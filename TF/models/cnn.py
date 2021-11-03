from tensorflow import keras
from tensorflow.keras import layers


class ConvModel(keras.Model):

    def __init__(self, input_shape):
        super().__init__()
        self.input_image_shape = input_shape
        self.model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(),
            layers.Conv2D(64, 5),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(),
            layers.Conv2D(128, 3, kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.5),
            layers.Dense(10)
        ])

    def call(self, inputs):
        return self.model(inputs)


class ResNet(keras.Model):
    class ResBlock(keras.Model):
        def __init__(self, filters: tuple, kernel_sizes: tuple, strides: tuple = ((1, 1), (1, 1)), activations: tuple = ('relu', 'relu'), paddings: tuple = ('same', 'same'), dilation_rates: tuple = ((1, 1), (1, 1))):
            super().__init__()

            self.model = keras.Sequential()
            self.build_res_block(filters=filters, kernel_sizes=kernel_sizes, strides=strides, activations=activations, paddings=paddings, dilation_rates=dilation_rates)

            # - Shortcut
            self.skip_connection_layer = layers.Add()
            self.identity_mapping_layer = layers.Conv2D(filters[1], 1, padding='same')

            # - Activation
            self.activation = layers.Activation(activations[1])

        def build_res_block(self, filters: tuple, kernel_sizes: tuple, strides: tuple, activations: tuple, paddings: tuple, dilation_rates: tuple):
            # I) - First conv block
            self.model.add(layers.Conv2D(filters[0], kernel_sizes[0], strides=strides[0], activation=activations[0], padding=paddings[0], dilation_rate=dilation_rates[0]))
            self.model.add(layers.BatchNormalization())
            # II) - Second conv block
            self.model.add(layers.Conv2D(filters[1], kernel_sizes[1], strides=strides[1], activation=None, padding=paddings[1], dilation_rate=dilation_rates[1]))
            self.model.add(layers.BatchNormalization())
            # III) - Output conv layer
            self.model.add(layers.Conv2D(filters[1], 1, padding='same'))

        def call(self, inputs, training=False):
            X = self.model(inputs)
            # > Skip connection
            # Depending on the output shape we'd use:
            # - input of the same number of channels if they are equal
            if X.shape[1:] == inputs.shape[1:]:
                X = self.skip_connection_layer(
                    [X, inputs]
                )
            # - perform a 1X1 convolution to increase the number of
            # channels to suit the output of the last Conv layer
            else:
                X = self.skip_connection_layer(
                    [X, self.identity_mapping_layer(inputs)]
                )

            return self.activation(X)

    def __init__(self, net_configs: dict):
        super().__init__()
        self.net_configs = net_configs
        self.model = keras.Sequential()
        self.build_net()

    def build_net(self):
        # 1) Input layer
        self.model.add(keras.Input(shape=self.net_configs.get('input_image_shape')))
        self.model.add(layers.Conv2D(**self.net_configs.get('conv2d_1')))
        self.model.add(layers.Conv2D(**self.net_configs.get('conv2d_2')))
        self.model.add(layers.MaxPool2D(**self.net_configs.get('max_pool_2d')))

        # 2) ResBlocks
        res_blocks_configs = self.net_configs.get('res_blocks')

        conv2_block_configs = res_blocks_configs.get('conv2_block_configs')
        for idx in range(conv2_block_configs.get('n_blocks')):
            self.model.add(self.ResBlock(**conv2_block_configs.get('block_configs')))

        conv3_block_configs = res_blocks_configs.get('conv3_block_configs')
        for idx in range(conv3_block_configs.get('n_blocks')):
            self.model.add(self.ResBlock(**conv3_block_configs.get('block_configs')))

        conv4_block_configs = res_blocks_configs.get('conv4_block_configs')
        for idx in range(conv4_block_configs.get('n_blocks')):
            self.model.add(self.ResBlock(**conv4_block_configs.get('block_configs')))

        conv5_block_configs = res_blocks_configs.get('conv5_block_configs')
        for idx in range(conv5_block_configs.get('n_blocks')):
            self.model.add(self.ResBlock(**conv5_block_configs.get('block_configs')))

        self.model.add(layers.Conv2D(**self.net_configs.get('conv2d_3')))

        self.model.add(layers.GlobalAveragePooling2D())

        self.model.add(layers.Dense(**self.net_configs.get('dense_layer')))

        self.model.add(layers.Dropout(**self.net_configs.get('dropout_layer')))

        self.model.add(layers.Dense(self.net_configs.get('number_of_classes')))

    def call(self, inputs, training=False):
        return self.model(inputs)

    def summary(self):
        return self.model.summary()
