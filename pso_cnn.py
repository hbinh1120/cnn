import tensorflow as tf
import random
import copy
import json
import os
import initializers

class ParticleSwarm:
    def __init__(self) -> None:
        self.g_best = None
        self.g_best_loss = float('inf')
        self.g_best_accuracy = 0
        self.particles = []

    def add_particles(self, num_particles):
        for _ in range(num_particles):
            particle = Particle()
            particle.initialize()
            self.particles.append(particle)

    def save_g_best(self, model_num):
        if not os.path.exists('save'):
            os.makedirs('save')
        with open('save/g_best.txt', 'w') as f:
            f.write(json.dumps(self.g_best.cnn_layers) + '\n')
            f.write(json.dumps(self.g_best.fc_layers) + '\n')
        with open('save/log.txt', 'a') as f:
            f.write(str(model_num) + ' ' + str(self.g_best_loss) + '\n')

    def load(self):
        if os.path.exists('save'):
            for filename in os.listdir('save'):
                with open(os.path.join('save/', filename), 'r') as f:
                    particle = Particle()
                    particle.cnn_layers = json.loads(f.readline())
                    particle.fc_layers = json.loads(f.readline())
                    self.particles.append(particle)

    def fit(self, x, y, tpu_strategy, w=.6, cg=.4, load=False, num_particles=10, iters=40, epochs=100, min_delta=0, patience=5):
        if load:
            self.load()
        else:
            self.add_particles(num_particles)

        for i in range(iters):
            print("------------------- ITERATION {:2d} -------------------".format(i + 1))

            #train all particles and update g_best
            for j, particle in enumerate(self.particles):
                print("PARTICLE {:2d}".format(j + 1), end=" ")
                if self.g_best is not None and self.g_best.fc_layers == particle.fc_layers and self.g_best.cnn_layers == particle.cnn_layers:
                    print("- particle is same as g_best and p_best")
                    particle.p_best = particle.copy()
                    particle.p_best_loss = self.g_best_loss
                    continue

                particle.save('particle' + str(j + 1) +'.txt')
                loss, model = particle.fit(x, y, tpu_strategy, epochs=epochs, min_delta=min_delta, patience=patience)
                if loss < particle.p_best_loss:
                    particle.p_best = particle.copy()
                    particle.p_best_loss = loss
                if loss < self.g_best_loss:
                    self.g_best = particle.copy()
                    self.g_best_loss = loss
                    print("New best model found")
                    self.save_g_best(i * num_particles + j + 1)
                    model.save('g_best.h5')

            #print gbest at the end of each iteration
            print("Best loss: ", self.g_best_loss)

            #update particles
            for particle in self.particles:
                particle.update(self.g_best, w, cg)

class Particle:
    def __init__(self) -> None:
        self.p_best = self
        self.p_best_loss = float('inf')

        self.cnn_layers = [] #both convolution layers and pooling layers
        self.fc_layers = [] #dense layers

    def initialize(self, initializer=initializers.sequential):
        self.cnn_layers, self.fc_layers = initializer()

    def save(self, filename):
        if not os.path.exists('save'):
            os.makedirs('save')
        with open('save/' + filename, 'w') as f:
            f.write(json.dumps(self.cnn_layers) + '\n')
            f.write(json.dumps(self.fc_layers) + '\n')

    def compile(self, tpu_strategy):
        #returns tf.keras.models.Sequential object
        layers = []

        #CNN layers, first layer is always convolution
        layers.append(tf.keras.layers.Conv2D(self.cnn_layers[0]['filters'], self.cnn_layers[0]['kernel_size'], padding='same', activation='relu', input_shape=(None, None, 3)))

        for layer in self.cnn_layers[1:]:
            if layer['type'] == 'dropout':
                layers.append(tf.keras.layers.Dropout(layer['rate']))
            elif layer['type'] == 'batchnorm':
                layers.append(tf.keras.layers.BatchNormalization())
            elif layer['type'] == 'conv':
                layers.append(tf.keras.layers.Conv2D(layer['filters'], layer['kernel_size'], padding='same', activation='relu'))
            elif layer['type'] == 'maxpool':
                layers.append(tf.keras.layers.MaxPooling2D(padding='same'))
            elif layer['type'] == 'avgpool':
                layers.append(tf.keras.layers.AveragePooling2D(padding='same'))

        #FC layers, global pooling instead of flatten to work with variable input size
        layers.append(tf.keras.layers.GlobalAveragePooling2D())
        for layer in self.fc_layers:
            if layer['type'] == 'dropout':
                layers.append(tf.keras.layers.Dropout(layer['rate']))
            elif layer['type'] == 'batchnorm':
                layers.append(tf.keras.layers.BatchNormalization())
            elif layer['type'] == 'fc':
                layers.append(tf.keras.layers.Dense(layer['units'], activation='relu'))

        #output layer, 10 outputs
        layers.append(tf.keras.layers.Dense(10, activation='softmax'))

        with tpu_strategy.scope():
            model = tf.keras.models.Sequential(layers)
            model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model

    def update(self, g_best, w, cg):
        self.update_fc(g_best, w, cg)
        self.update_cnn(g_best, w, cg)

    def update_fc(self, g_best, w, cg):
        new_fc_layers = []
        for i in range(max(len(g_best.fc_layers), len(self.fc_layers), len(self.p_best.fc_layers))):
            if random.random() < w:
                if i < len(self.fc_layers):
                    new_fc_layers.append(self.fc_layers[i])
            else:
                if random.random() < cg:
                    if i < len(g_best.fc_layers):
                        new_fc_layers.append(g_best.fc_layers[i])
                else:
                    if i < len(self.p_best.fc_layers):
                        new_fc_layers.append(self.p_best.fc_layers[i])
        self.fc_layers = new_fc_layers

    def update_cnn(self, g_best, w, cg):
        new_cnn_layers = []
        for i in range(max(len(g_best.cnn_layers), len(self.cnn_layers), len(self.p_best.cnn_layers))):
            if random.random() < w:
                if i < len(self.cnn_layers):
                    new_cnn_layers.append(self.cnn_layers[i])
            else:
                if random.random() < cg:
                    if i < len(g_best.cnn_layers):
                        new_cnn_layers.append(g_best.cnn_layers[i])
                else:
                    if i < len(self.p_best.cnn_layers):
                        new_cnn_layers.append(self.p_best.cnn_layers[i])
        self.cnn_layers = new_cnn_layers


    def copy(self):
        particle = Particle()
        for layer in self.fc_layers:
            particle.fc_layers.append(copy.deepcopy(layer))
        for layer in self.cnn_layers:
            particle.cnn_layers.append(copy.deepcopy(layer))
        return particle

    def fit(self, x, y, tpu_strategy, epochs=100, batch_size=256, min_delta=0, patience=5):
        model = self.compile(tpu_strategy)
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, restore_best_weights=True)
        history = model.fit(x, y, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[earlystop], verbose=0)

        index_min = history.history["val_loss"].index(min(history.history["val_loss"]))
        loss = history.history["loss"][index_min]
        accuracy = history.history["accuracy"][index_min]
        val_loss = history.history["val_loss"][index_min]
        val_accuracy = history.history["val_accuracy"][index_min]
        print("- loss: {:1.4f} - accuracy: {:1.4f} - val_loss: {:1.4f} - val_accuracy: {:1.4f}".format(loss, accuracy, val_loss, val_accuracy))

        return val_loss, model

if __name__ == '__main__':
    #loads dataset, tf.keras.datasets.<x> where <x> is the name of the dataset
    dataset = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train, x_test = x_train / 255., x_test / 255.

    #tpu stuff
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        raise BaseException('ERROR: Not connected to a TPU runtime')

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    #particle swarm to find best model
    swarm = ParticleSwarm()
    swarm.fit(x_train, y_train, tpu_strategy)