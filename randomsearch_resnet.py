import tensorflow as tf
import os
import json
from initializers import Initializers

class Model:
    def __init__(self) -> None:
        self.best_loss = float('inf')
        self.nodes = []

    def initialize(self, initializer=Initializers()):
        self.initializer = initializer
        self.nodes = self.initializer.cell()

    def save_best(self, model_num):
        if not os.path.exists('save'):
            os.makedirs('save')
        with open('save/best.txt', 'w') as f:
            f.write(json.dumps(self.nodes) + '\n')
        with open('save/log.txt', 'a') as f:
            f.write(str(model_num) + ' ' + str(self.best_loss) + '\n')

    def compile(self, tpu_strategy, motif_repeats=3, cell_repeats=3):
        #returns a tf model
        with tpu_strategy.scope():
            inputs = tf.keras.Input(shape=(None, None, 3))
            model_graph = []
            model_graph.append(inputs)
            model_graph.append(inputs)

            for repeat in range(motif_repeats):
                for _ in range(cell_repeats):
                    node_outputs = []
                    node_outputs.append(model_graph[-1])
                    node_outputs.append(model_graph[-2])
                    used_nodes = []
                    for node in self.nodes:
                        node_inputs = []
                        for i in range(len(node['inputs'])):
                            used_nodes.append(node['inputs'][i])
                            if node['operations'][i]['type'] == 'conv':
                                output = tf.keras.layers.Conv2D(node['operations'][i]['filters'], node['operations'][i]['kernel_size'], padding='same')(node_outputs[node['inputs'][i] + 2])
                                output = tf.keras.layers.BatchNormalization()(output)
                                output = tf.keras.layers.Activation(tf.keras.activations.relu)(output)
                                node_inputs.append(output)
                            elif node['operations'][i]['type'] == 'maxpool':
                                node_inputs.append(tf.keras.layers.MaxPooling2D(padding='same')(node_outputs[node['inputs'][i] + 2]))
                            elif node['operations'][i]['type'] == 'avgpool':
                                node_inputs.append(tf.keras.layers.AveragePooling2D(padding='same')(node_outputs[node['inputs'][i] + 2]))
                            elif node['operations'][i]['type'] == 'none':
                                node_inputs.append(node_outputs[node['inputs'][i] + 2])
                        
                        #get largest dimensions
                        max_channel = 0
                        for output in node_inputs:
                            dimensions = output.shape.as_list()
                            if dimensions[3] > max_channel:
                                max_channel = dimensions[3]

                        #combine all outputs
                        if node['combine_method'] == 'add':
                            #adds more channels to match inputs
                            for i, output in enumerate(node_inputs):
                                output = tf.keras.layers.Conv2D(max_channel, 1, padding='same', activation='relu')(output)
                                output = tf.keras.layers.BatchNormalization()(output)
                                output = tf.keras.layers.Activation(tf.keras.activations.relu)(output)
                                node_inputs[i] = output
                            node_outputs.append(tf.keras.layers.Add()(node_inputs))
                        elif node['combine_method'] == 'concatenate':
                            node_outputs.append(tf.keras.layers.Concatenate()(node_inputs))

                    #point all nodes that dont have an output to the output node
                    nodes_to_outputs = []
                    for i, node in enumerate(node_outputs):
                        if i not in used_nodes:
                            nodes_to_outputs.append(node)
                    cell_outputs = tf.keras.layers.Concatenate()(nodes_to_outputs)

                if repeat < motif_repeats - 1:
                    model_graph.append(tf.keras.layers.AveragePooling2D()(model_graph[-1]))
                    cell_outputs = tf.keras.layers.AveragePooling2D()(cell_outputs)
                model_graph.append(cell_outputs)

            #global pooling instead of flatten to work with variable input size
            model_outputs = tf.keras.layers.GlobalAveragePooling2D()(model_graph[-1])
            #output layer, 10 outputs
            model_outputs = tf.keras.layers.Dense(10, activation='softmax')(model_outputs)

            model = tf.keras.Model(inputs=inputs, outputs=model_outputs, name='resnet')
            model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model

    def fit(self, x, y, tpu_strategy, iters=300, epochs=20, batch_size=128, min_delta=0, patience=5):
        initializer = Initializers()
        for i in range(iters):
            print("MODEL {}".format(i + 1), end=" ")
            self.initialize(initializer)
            model = self.compile(tpu_strategy)
            earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, restore_best_weights=True)
            history = model.fit(x, y, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[earlystop], verbose=0)

            index_min = history.history["val_loss"].index(min(history.history["val_loss"]))
            loss = history.history["loss"][index_min]
            accuracy = history.history["accuracy"][index_min]
            val_loss = history.history["val_loss"][index_min]
            val_accuracy = history.history["val_accuracy"][index_min]
            print("- loss: {:1.4f} - accuracy: {:1.4f} - val_loss: {:1.4f} - val_accuracy: {:1.4f}".format(loss, accuracy, val_loss, val_accuracy))

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print("New best model found")
                self.save_best(i)
                model.save('best.h5')

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
    model = Model()
    model.fit(x_train, y_train, tpu_strategy)