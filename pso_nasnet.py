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

    def fit(self, x, y, tpu_strategy, w=.95, c1=4, c2=4, load=False, num_particles=10, iters=40, epochs=100, min_delta=0, patience=5):
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
                    tf.keras.utils.plot_model(model, "g_best.png")

            #print gbest at the end of each iteration
            print("Best loss: ", self.g_best_loss)

            #update particles
            for particle in self.particles:
                particle.update_velocity(self.g_best, w, c1, c2)
                particle.update_position()

class Particle:
    def __init__(self) -> None:
        self.p_best = self
        self.p_best_loss = float('inf')
        self.velocity = []

        self.nodes = []

    def initialize(self, initializer=Initializers()):
        self.initializer = initializer
        self.nodes = self.initializer.cell()
        for node in self.nodes:
            node_velocity = []
            for _ in range(len(node['graph'])):
                node_velocity.append(-4)
            self.velocity.append(node_velocity)

    def save(self, filename):
        if not os.path.exists('save'):
            os.makedirs('save')
        with open('save/' + filename, 'w') as f:
            f.write(json.dumps(self.nodes) + '\n')
            f.write(json.dumps(self.p_best.nodes) + '\n')

    def compile(self, tpu_strategy, reductions=3, cell_repeats=2, filters=16):
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
                    used_nodes = []
                    for node_index, node in enumerate(self.nodes):
                        node_inputs = []
                        for i, edge in enumerate(node['graph']):
                            if edge == 1:
                                used_nodes.append(node['edges'][i]['from_node'])
                                if node['edges'][i]['type'] == 'conv':
                                    output = tf.keras.layers.BatchNormalization()(node_outputs[node['edges'][i]['from_node']])
                                    output = tf.keras.layers.Activation(tf.keras.activations.relu)(output)
                                    output = tf.keras.layers.Conv2D(filters * (2 ** reduction), node['edges'][i]['kernel_size'], padding='same')(output)
                                    node_inputs.append(output)
                                elif node['edges'][i]['type'] == 'separableconv':
                                    output = tf.keras.layers.BatchNormalization()(node_outputs[node['edges'][i]['from_node']])
                                    output = tf.keras.layers.Activation(tf.keras.activations.relu)(output)
                                    output = tf.keras.layers.SeparableConv2D(filters * (2 ** reduction), node['edges'][i]['kernel_size'], padding='same')(output)
                                    node_inputs.append(output)
                                elif node['edges'][i]['type'] == 'none':
                                    node_inputs.append(node_outputs[node['edges'][i]['from_node']])
                        
                        #dont use node if it has no input
                        if len(node_inputs) == 0:
                            used_nodes.append(node_index + 2)
                            node_outputs.append(node_outputs[0])
                            continue

                        #get largest dimensions
                        max_channel = 0
                        for output in node_inputs:
                            dimensions = output.shape.as_list()
                            if dimensions[3] > max_channel:
                                max_channel = dimensions[3]

                        #combine inputs of each node into its output
                        if node['combine_method'] == 'add':
                            #adds more channels to match inputs
                            for i, output in enumerate(node_inputs):
                                if output.shape.as_list()[3] != max_channel:
                                    output = tf.keras.layers.BatchNormalization()(output)
                                    output = tf.keras.layers.Activation(tf.keras.activations.relu)(output)
                                    output = tf.keras.layers.Conv2D(max_channel, 1, padding='same')(output)
                                    node_inputs[i] = output
                            if len(node_inputs) > 1:
                                node_outputs.append(tf.keras.layers.Add()(node_inputs))
                            else:
                                node_outputs.append(node_inputs[0])
                        elif node['combine_method'] == 'concatenate':
                            if len(node_inputs) > 1:
                                node_outputs.append(tf.keras.layers.Concatenate()(node_inputs))
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

                #reduce dimensions of inputs for next repeat
                channels = input_nodes[0].shape.as_list()[-1]
                input_nodes[0] = tf.keras.layers.BatchNormalization()(input_nodes[0])
                input_nodes[0] = tf.keras.layers.Activation(tf.keras.activations.relu)(input_nodes[0])
                input_nodes[0] = tf.keras.layers.Conv2D(int(0.5 * channels), 1, padding='same')(input_nodes[0])
                input_nodes[0] = tf.keras.layers.AveragePooling2D()(input_nodes[0])
   
                channels = input_nodes[1].shape.as_list()[-1]
                input_nodes[1] = tf.keras.layers.BatchNormalization()(input_nodes[1])
                input_nodes[1] = tf.keras.layers.Activation(tf.keras.activations.relu)(input_nodes[1])
                input_nodes[1] = tf.keras.layers.Conv2D(int(0.5 * channels), 1, padding='same')(input_nodes[1])
                input_nodes[1] = tf.keras.layers.AveragePooling2D()(input_nodes[1])

            model_outputs = tf.keras.layers.Concatenate()(input_nodes)
            #global pooling instead of flatten to work with variable input size
            model_outputs = tf.keras.layers.GlobalAveragePooling2D()(model_outputs)
            #output layer, 10 outputs
            model_outputs = tf.keras.layers.Dense(10, activation='softmax')(model_outputs)

            #make model out of these layers
            model = tf.keras.Model(inputs=inputs, outputs=model_outputs, name='nasnet')
            model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model

    def update_velocity(self, g_best, w, c1, c2):
        for i, node_velocity in enumerate(self.velocity):
            for j, _ in enumerate(node_velocity):
                self.velocity[i][j] = w * self.velocity[i][j] \
                    + c1 * random.random() * (self.p_best.nodes[i]['graph'][j] - self.nodes[i]['graph'][j]) \
                    + c2 * random.random() * (g_best.nodes[i]['graph'][j] - self.nodes[i]['graph'][j])
                if self.velocity[i][j] > 4:
                    self.velocity[i][j] = 4
                elif self.velocity[i][j] < -4:
                    self.velocity[i][j] = -4

    def update_position(self):
        for i, node_velocity in enumerate(self.velocity):
            for j, dim_velocity in enumerate(node_velocity):
                if random.random() >= 1 / (1 + exp(-dim_velocity)):
                    self.nodes[i]['graph'][j] = 0
                else:
                    self.nodes[i]['graph'][j] = 1

    def copy(self):
        particle = Particle()
        for layer in self.nodes:
            particle.nodes.append(copy.deepcopy(layer))
        return particle

    def fit(self, x, y, tpu_strategy, epochs=100, batch_size=128, min_delta=0, patience=5):
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
    tpu_strategy = tf.distribute.TPUStrategy(tpu)

    #particle swarm to find best model
    swarm = ParticleSwarm()
    swarm.fit(x_train, y_train, tpu_strategy)