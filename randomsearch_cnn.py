import tensorflow as tf
import initializers
import os
import json

class Model:
    def __init__(self) -> None:
        self.best_loss = float('inf')
        self.cnn_layers = [] #both convolution layers and pooling layers
        self.fc_layers = [] #dense layers

    def initialize(self, initializer=initializers.Initializers()):
        self.cnn_layers, self.fc_layers = initializer.sequential()

    def save_best(self, model_num):
        if not os.path.exists('save'):
            os.makedirs('save')
        with open('save/best.txt', 'w') as f:
            f.write(json.dumps(self.cnn_layers) + '\n')
            f.write(json.dumps(self.fc_layers) + '\n')

        with open('save/log.txt', 'a') as f:
            f.write(str(model_num) + ' ' + str(self.best_loss) + '\n')

    def compile(self, tpu_strategy):
        #returns tf.keras.models.Sequential object
        layers = []

        #CNN layers, first layer is always convolution
        layers.append(tf.keras.layers.Conv2D(self.cnn_layers[0]['filters'], self.cnn_layers[0]['kernel_size'], padding='same', activation='relu', input_shape=(None, None, 3)))

        for layer in self.cnn_layers[1:]:
            if layer['type'] == 'conv':
                layers.append(tf.keras.layers.Conv2D(layer['filters'], layer['kernel_size'], padding='same'))
                layers.append(tf.keras.layers.BatchNormalization())
                layers.append(tf.keras.layers.Activation(tf.keras.activations.relu))
            elif layer['type'] == 'maxpool':
                layers.append(tf.keras.layers.MaxPooling2D(padding='same'))
            elif layer['type'] == 'avgpool':
                layers.append(tf.keras.layers.AveragePooling2D(padding='same'))

        #FC layers, global pooling instead of flatten to work with variable input size
        layers.append(tf.keras.layers.GlobalAveragePooling2D())
        for layer in self.fc_layers:
            if layer['type'] == 'fc':
                layers.append(tf.keras.layers.Dense(layer['units']))
                layers.append(tf.keras.layers.BatchNormalization())
                layers.append(tf.keras.layers.Activation(tf.keras.activations.relu))

        #output layer, 10 outputs
        layers.append(tf.keras.layers.Dense(10, activation='softmax'))

        with tpu_strategy.scope():
            model = tf.keras.models.Sequential(layers)
            model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model

    def fit(self, x, y, tpu_strategy, iters=300, epochs=100, batch_size=1024, min_delta=0, patience=5):
        for i in range(iters):
            print("MODEL {}".format(i + 1), end=" ")
            self.initialize()
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
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    #particle swarm to find best model
    model = Model()
    model.fit(x_train, y_train, tpu_strategy)