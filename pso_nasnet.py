from math import exp
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
        for _ in range(num_particles):
            particle = Particle()
            particle.initialize()
            self.particles.append(particle)

    def save_g_best(self, model_num):
        if not os.path.exists('save'):
            os.makedirs('save')
        with open('save/g_best.txt', 'w') as f:
            f.write(json.dumps(self.g_best.position) + '\n')
            f.write(json.dumps(self.g_best_loss) + '\n')
        if model_num == 1:
            with open('save/log.txt', 'w') as f:
                f.write(str(model_num) + ' ' + str(self.g_best_loss) + '\n')
        else:
            with open('save/log.txt', 'a') as f:
                f.write(str(model_num) + ' ' + str(self.g_best_loss) + '\n')

    def load(self, num_particles=5):
        self.g_best = Particle()
        with open('save/g_best.txt', 'r') as f:
            self.g_best.position = json.loads(f.readline())
            self.g_best_loss = float(f.readline())
        for i in range(num_particles):
            filename = 'particle' + str(i + 1) + '.txt'
            particle = Particle()
            particle.load(filename)
            self.particles.append(particle)

    def fit(self, x, y, tpu_strategy, w=1, c1=4, c2=4, load=False, num_particles=5, iters=40, epochs=300, min_delta=0, patience=10):
        if load:
            self.load(num_particles)
        else:
            self.add_particles(num_particles)

        for i in range(iters):
            print("------------------- ITERATION {:2d} -------------------".format(i + 1))

            #save all particles
            for j, particle in enumerate(self.particles):
                particle.save('particle' + str(j + 1) +'.txt')
            
            #train all particles and update g_best
            for j, particle in enumerate(self.particles):
                print("PARTICLE {:2d}".format(j + 1), end=" ")        
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
                    tf.keras.utils.plot_model(model, "g_best.png")

            #print gbest at the end of each iteration
            print("Best loss: ", self.g_best_loss)

            #update particles
            for particle in self.particles:
                particle.update_velocity(self.g_best, w, c1, c2)
                particle.update_position()

class Particle:
    initializer = Initializers()
    graph = initializer.cell_graph()

    def __init__(self) -> None:
        self.p_best = self
        self.p_best_loss = float('inf')
        self.velocity = []
        self.position = []

    def initialize(self):
        self.position = Particle.initializer.cell_init(Particle.graph)
        for node in self.position:
            node_velocity = []
            for _ in range(len(node)):
                node_velocity.append(-4)
            self.velocity.append(node_velocity)

    def save(self, filename):
        if not os.path.exists('save'):
            os.makedirs('save')
        with open('save/' + filename, 'w') as f:
            f.write(json.dumps(self.position) + '\n')
            f.write(json.dumps(self.p_best.position) + '\n')
            f.write(json.dumps(self.velocity) + '\n')
            f.write(json.dumps(self.p_best_loss) + '\n')

    def load(self, filename):
        if not os.path.exists('save/' + filename):
            self.initialize()
        else:
            with open('save/' + filename, 'r') as f:
                self.position = json.loads(f.readline())
                self.p_best = Particle()
                self.p_best.position = json.loads(f.readline())
                self.velocity = json.loads(f.readline())
                self.p_best_loss = float(f.readline())

    def compile(self, tpu_strategy, reductions=3, cell_repeats=2, filters=24):
        #returns a tf model
        with tpu_strategy.scope():
            #pair of inputs to use for each cell
            inputs = tf.keras.Input(shape=(None, None, 3))
            input_nodes = [inputs, inputs]

            for reduction in range(reductions):
                for _ in range(cell_repeats):
                    #list of outputs of each node in cell
                    node_outputs = []
                    node_outputs.append(input_nodes[0])
                    node_outputs.append(input_nodes[1])
                    used_nodes = {}
                    for node_index, node in enumerate(self.position):
                        node_inputs = []
                        for i, edge in enumerate(node):
                            if edge == 1:
                                used_nodes[Particle.graph[node_index][i]['from_node']] = 1
                                output = node_outputs[Particle.graph[node_index][i]['from_node']]
                                if Particle.graph[node_index][i]['type'] == 'conv':
                                    output = tf.keras.layers.Conv2D(filters * (2 ** reduction), Particle.graph[node_index][i]['kernel_size'], padding='same')(output)
                                    output = tf.keras.layers.BatchNormalization()(output)
                                    output = tf.keras.layers.Activation(tf.keras.activations.relu)(output)
                                elif Particle.graph[node_index][i]['type'] == 'separableconv':
                                    output = tf.keras.layers.SeparableConv2D(filters * (2 ** reduction), Particle.graph[node_index][i]['kernel_size'], padding='same')(output)
                                    output = tf.keras.layers.BatchNormalization()(output)
                                    output = tf.keras.layers.Activation(tf.keras.activations.relu)(output)
                                elif Particle.graph[node_index][i]['type'] == 'maxpool':
                                    output = tf.keras.layers.MaxPooling2D(pool_size=Particle.graph[node_index][i]['pool_size'], strides=Particle.graph[node_index][i]['stride'], padding='same')(output)
                                elif Particle.graph[node_index][i]['type'] == 'averagepool':
                                    output = tf.keras.layers.AveragePooling2D(pool_size=Particle.graph[node_index][i]['pool_size'], strides=Particle.graph[node_index][i]['stride'], padding='same')(output)
                                elif Particle.graph[node_index][i]['type'] == 'none':
                                    pass
                                output = tf.keras.layers.Dropout(0.2, (None, 1, 1, 1))(output)
                                node_inputs.append(output)
                        
                        #dont use node if it has no input
                        if len(node_inputs) == 0:
                            used_nodes[node_index + 2] = 1
                            node_outputs.append(node_outputs[0])
                            continue

                        #get largest dimensions
                        max_channel = 0
                        for output in node_inputs:
                            dimensions = output.shape.as_list()
                            if dimensions[3] > max_channel:
                                max_channel = dimensions[3]

                        #combine inputs of each node into its output
                        #adds more channels to match inputs
                        for i, output in enumerate(node_inputs):
                            if output.shape.as_list()[3] != max_channel:
                                output = tf.keras.layers.Conv2D(max_channel, 1, padding='same')(output)
                                output = tf.keras.layers.BatchNormalization()(output)
                                output = tf.keras.layers.Activation(tf.keras.activations.relu)(output)
                                node_inputs[i] = output
                        if len(node_inputs) > 1:
                            node_outputs.append(tf.keras.layers.Add()(node_inputs))
                        else:
                            node_outputs.append(node_inputs[0])

                    #point all nodes that dont have an output to the output node
                    nodes_to_outputs = []
                    for i, node in enumerate(node_outputs):
                        if i not in used_nodes:
                            nodes_to_outputs.append(node)
                    cell_outputs = tf.keras.layers.Concatenate()(nodes_to_outputs)
                    
                    #update inputs for next cell
                    input_nodes[0] = input_nodes[1]
                    input_nodes[1] = cell_outputs

                #reduce dimensions of inputs if not last repeat
                if reduction < reductions - 1:
                    channels = input_nodes[0].shape.as_list()[-1]
                    input_nodes[0] = tf.keras.layers.Conv2D(filters=int(.5 * channels), kernel_size=1, strides=2, padding='same')(input_nodes[0])
                    input_nodes[0] = tf.keras.layers.BatchNormalization()(input_nodes[0])
                    input_nodes[0] = tf.keras.layers.Activation(tf.keras.activations.relu)(input_nodes[0])
    
                    channels = input_nodes[1].shape.as_list()[-1]
                    input_nodes[1] = tf.keras.layers.Conv2D(filters=int(.5 * channels), kernel_size=1, strides=2, padding='same')(input_nodes[1])
                    input_nodes[1] = tf.keras.layers.BatchNormalization()(input_nodes[1])
                    input_nodes[1] = tf.keras.layers.Activation(tf.keras.activations.relu)(input_nodes[1])

            #global pooling instead of flatten to work with variable input size
            model_outputs = tf.keras.layers.GlobalAveragePooling2D()(input_nodes[1])
            #output layer, 10 outputs
            model_outputs = tf.keras.layers.Dense(10, activation='softmax')(model_outputs)

            #make model out of these layers
            model = tf.keras.Model(inputs=inputs, outputs=model_outputs, name='nasnet')
            model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model

    def update_velocity(self, g_best, w, c1, c2):
        for i, node_velocity in enumerate(self.velocity):
            for j, _ in enumerate(node_velocity):
                self.velocity[i][j] = w * self.velocity[i][j] \
                    + c1 * random.random() * (self.p_best.position[i][j] - self.position[i][j]) \
                    + c2 * random.random() * (g_best.position[i][j] - self.position[i][j])
                if self.velocity[i][j] > 4:
                    self.velocity[i][j] = 4
                elif self.velocity[i][j] < -4:
                    self.velocity[i][j] = -4

    def update_position(self):
        for i, node_velocity in enumerate(self.velocity):
            for j, dim_velocity in enumerate(node_velocity):
                if random.random() >= 1 / (1 + exp(-dim_velocity)):
                    self.position[i][j] = 0
                else:
                    self.position[i][j] = 1

    def copy(self):
        particle = Particle()
        for layer in self.position:
            particle.position.append(copy.deepcopy(layer))
        return particle

    def fit(self, x, y, tpu_strategy, epochs=300, batch_size=128, min_delta=0, patience=10):
        model = self.compile(tpu_strategy)
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, restore_best_weights=True)
        history = model.fit(x, y, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[earlystop], verbose=0)

        index_min = history.history["val_loss"].index(min(history.history["val_loss"]))
        loss = history.history["loss"][index_min]
        accuracy = history.history["accuracy"][index_min]
        val_loss = history.history["val_loss"][index_min]
        val_accuracy = history.history["val_accuracy"][index_min]
        print("- epoch: {:3d} - loss: {:1.4f} - accuracy: {:1.4f} - val_loss: {:1.4f} - val_accuracy: {:1.4f}".format(index_min + 1, loss, accuracy, val_loss, val_accuracy))

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
    tpu_strategy = tf.distribute.TPUStrategy(tpu)

    #particle swarm to find best model
    swarm = ParticleSwarm()
    swarm.fit(x_train, y_train, tpu_strategy)