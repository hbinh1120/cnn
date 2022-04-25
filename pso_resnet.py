import tensorflow as tf
import random
import copy
import json
import os
from initializers import Initializers

class ParticleSwarm:
    def __init__(self) -> None:
        self.g_best = None
        self.g_best_loss = float('inf')
        self.particles = []

    def add_particles(self, num_particles):
        initializer = Initializers()
        for _ in range(num_particles):
            particle = Particle()
            particle.initialize(initializer)
            self.particles.append(particle)

    def save_g_best(self, model_num):
        if not os.path.exists('save'):
            os.makedirs('save')
        with open('save/g_best.txt', 'w') as f:
            f.write(json.dumps(self.g_best.nodes) + '\n')
        with open('save/log.txt', 'a') as f:
            f.write(str(model_num) + ' ' + str(self.g_best_loss) + '\n')

    def load(self):
        if os.path.exists('save'):
            for filename in os.listdir('save'):
                with open(os.path.join('save/', filename), 'r') as f:
                    particle = Particle()
                    particle.nodes = json.loads(f.readline())
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
                if self.g_best is not None and self.g_best.nodes == particle.nodes:
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

        self.nodes = []

    def initialize(self, initializer=Initializers()):
        self.initializer = initializer
        self.nodes = self.initializer.cell()

    def save(self, filename):
        if not os.path.exists('save'):
            os.makedirs('save')
        with open('save/' + filename, 'w') as f:
            f.write(json.dumps(self.nodes) + '\n')

    def compile(self, tpu_strategy):
        #returns a tf model
        with tpu_strategy.scope():
            graph = []
            inputs = tf.keras.Input(shape=(None, None, 3))
            graph.append(inputs)
            graph.append(inputs)
            for node in self.nodes:
                if node is None:
                    continue
                node_outputs = []
                for i in range(len(node['inputs'])):
                    if node['operations'][i]['type'] == 'conv':
                        node_outputs.append(tf.keras.layers.Conv2D(node['operations'][i]['filters'], node['operations'][i]['kernel_size'], padding='same', activation='relu')(graph[node['inputs'][i]]))
                    elif node['operations'][i]['type'] == 'maxpool':
                        node_outputs.append(tf.keras.layers.MaxPooling2D(padding='same')(graph[node['inputs'][i]]))
                    elif node['operations'][i]['type'] == 'avgpool':
                        node_outputs.append(tf.keras.layers.AveragePooling2D(padding='same')(graph[node['inputs'][i]]))
                    elif node['operations'][i]['type'] == 'none':
                        node_outputs.append(graph[node['inputs'][i]])
                
                #get largest dimensions
                max_channel = 0
                for output in node_outputs:
                    dimensions = output.shape.as_list()
                    if dimensions[3] > max_channel:
                        max_channel = dimensions[3]

                #resize all outputs to largest dimensions
                for i, output in enumerate(node_outputs):
                    node_outputs[i] = tf.keras.layers.Conv2D(max_channel, 1, padding='same', activation='relu')(output)

                #combine all outputs
                if node['combine_method'] == 'add':
                    graph.append(tf.keras.layers.Add()(node_outputs))
                elif node['combine_method'] == 'concatenate':
                    graph.append(tf.keras.layers.Concatenate()(node_outputs))

            #global pooling instead of flatten to work with variable input size
            outputs = tf.keras.layers.GlobalAveragePooling2D()(graph[-1])
            #output layer, 10 outputs
            outputs = tf.keras.layers.Dense(10, activation='softmax')(outputs)

            model = tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet')
            model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model

    def update(self, g_best, w, cg):
        new_nodes = []
        for i in range(max(len(g_best.nodes), len(self.nodes), len(self.p_best.nodes))):
            if random.random() < w:
                if i < len(self.nodes):
                    new_nodes.append(self.nodes[i])
            else:
                if random.random() < cg:
                    if i < len(g_best.nodes):
                        new_nodes.append(g_best.nodes[i])
                else:
                    if i < len(self.p_best.nodes):
                        new_nodes.append(self.p_best.nodes[i])
        for i, node in enumerate(new_nodes[:-1]):
            if node is not None:
                node['inputs'] = node['inputs'][:2]
                node['operations'] = node['operations'][:2]
                for j, input in enumerate(node['inputs']):
                    if input >= i:
                        node['inputs'][j] = random.choice(list(range(j)))
        unused_nodes = list(range(len(new_nodes) - 1))
        for node in new_nodes[:-1]:
            if node is not None:
                for input in node['inputs']:
                    try:
                        unused_nodes.remove(input)
                    except:
                        pass

        new_nodes[-1] = {}
        new_nodes[-1]['inputs'] = []
        new_nodes[-1]['operations'] = []
        new_nodes[-1]['combine_method'] = 'concatenate'
        for i in unused_nodes:
            op = {}
            type = random.choice(self.initializer.config['operations'])
            op['type'] = type
            for key in self.initializer.config['config'][type]:
                op[key] = random.choice(self.initializer.config['config'][type][key])
            new_nodes[-1]['inputs'].append(i)
            new_nodes[-1]['operations'].append(op)
        self.nodes = new_nodes


    def copy(self):
        particle = Particle()
        for layer in self.nodes:
            particle.nodes.append(copy.deepcopy(layer))
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