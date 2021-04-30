import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers import BatchNormalization


class SiameseNetwork:
    def __init__(self,
                 input_shape,
                 learning_rate=0.001,
                 momentum=0.5,
                 decay_rate=0.98,
                 use_sgd=True,
                 kernel_regularizer_conv=0.01,
                 kernel_regularizer_dense=0.0001):
        self._convolutional_network = Sequential()
        self._set_cnn(kernel_regularizer_conv, kernel_regularizer_dense)
        self.model = self._build_siamese_network(input_shape, learning_rate, momentum, decay_rate, use_sgd)

    def _add_convolutional_layer(self, filters, kernel_size, kernel_regularizer, add_pool):
        self._convolutional_network.add(Conv2D(filters=filters,
                                               kernel_size=kernel_size,
                                               activation='relu',
                                               kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                                               bias_initializer=RandomNormal(mean=0.5, stddev=0.01),
                                               kernel_regularizer=l2(kernel_regularizer)))
        self._convolutional_network.add(BatchNormalization())

        if add_pool:
            self._convolutional_network.add(MaxPool2D())

    def _set_cnn(self, kernel_regularizer_conv, kernel_regularizer_dense):
        self._add_convolutional_layer(64, (10, 10), kernel_regularizer_conv, True)
        self._add_convolutional_layer(128, (7, 7), kernel_regularizer_conv, True)
        self._add_convolutional_layer(128, (4, 4), kernel_regularizer_conv, True)
        self._add_convolutional_layer(256, (4, 4), kernel_regularizer_conv, False)

        self._convolutional_network.add(Flatten())
        self._convolutional_network.add(Dense(units=4096,
                                              activation='sigmoid',
                                              kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                                              bias_initializer=RandomNormal(mean=0.5, stddev=0.01),
                                              kernel_regularizer=l2(kernel_regularizer_dense)))

    def _build_siamese_network(self,
                               input_shape,
                               learning_rate=0.001,
                               momentum=0.5,
                               decay_rate=0.98,
                               use_sgd=True):

        input_image_1 = Input(input_shape)
        input_image_2 = Input(input_shape)

        output_image_1 = self._convolutional_network(input_image_1)
        output_image_2 = self._convolutional_network(input_image_2)

        l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        l1_distance = l1_distance_layer([output_image_1, output_image_2])

        prediction = Dense(units=1,
                           activation='sigmoid',
                           kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                           bias_initializer=RandomNormal(mean=0.5, stddev=0.01))(l1_distance)
        model = Model(inputs=[input_image_1, input_image_2], outputs=prediction)

        if use_sgd:
            optimizer = SGD(learning_rate=ExponentialDecay(learning_rate, 100000, decay_rate),
                            momentum=momentum)
        else:
            optimizer = Adam(learning_rate=ExponentialDecay(learning_rate, 100000, decay_rate))

        model.compile(loss='binary_crossentropy',
                      metrics=['binary_accuracy'],
                      optimizer=optimizer)
        return model

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, **kwargs)

    def evaluate(self, X, y, **kwargs):
        return self.model.evaluate(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)




