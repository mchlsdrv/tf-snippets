import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.nn import relu
from tensorflow.keras import layers, losses, optimizers


class ResNet(keras.Model):
    class ResBlock(layers.Layer):
        def __init__(self, filters: tuple, kernel_sizes: tuple, strides: tuple = ((1, 1), (1, 1)), activations: tuple = ('relu', 'relu'), paddings: tuple = ('same', 'same'), dilation_rates: tuple = ((1, 1), (1, 1))):
            super().__init__()

            # I) - First conv block
            self.conv2d_1 = layers.Conv2D(
                filters[0], kernel_sizes[0], strides=strides[0], activation=activations[0], padding=paddings[0], dilation_rate=dilation_rates[0])
            self.batch_norm_1 = layers.BatchNormalization()

            # II) - Second conv block
            self.conv2d_2 = layers.Conv2D(
                filters[1], kernel_sizes[1], strides=strides[1], activation=None, padding=paddings[1], dilation_rate=dilation_rates[1])
            self.batch_norm_2 = layers.BatchNormalization()

            # III) - Skip connection
            self.identity = layers.Conv2D(filters[1], 1, padding='same')
            self.shortcut = layers.Add()

            # IV) - Activation
            self.activation = layers.Activation(activations[1])

        def call(self, inputs, training=False):
            x = self.conv2d_1(inputs)
            x = self.batch_norm_1(x)
            x = self.conv2d_2(x)
            x = self.batch_norm_2(x)

            if x.shape[1:] == inputs.shape[1:]:
                x = self.shortcut([x, inputs])
            else:
                x = self.shortcut([x, self.identity(inputs)])

            return self.activation(x)

    def __init__(self, net_configs: dict):
        super().__init__()
        self.input_image_shape = net_configs.get('input_image_shape')
        # 1) Input layer
        self.input_layer = keras.Input(shape=self.input_image_shape)

        self.conv2d_1 = layers.Conv2D(**net_configs.get('conv2d_1'))

        self.conv2d_2 = layers.Conv2D(**net_configs.get('conv2d_2'))

        self.max_pool2d = layers.MaxPool2D(**net_configs.get('max_pool_2d'))


        # 2) ResBlocks
        res_blocks_configs = net_configs.get('res_blocks')

        conv2_block_configs = res_blocks_configs.get('conv2_block_configs')
        self.conv2_blocks = []
        for idx in range(conv2_block_configs.get('n_blocks')):
            self.conv2_blocks.append(self.ResBlock(**conv2_block_configs.get('block_configs')))

        conv3_block_configs = res_blocks_configs.get('conv3_block_configs')
        self.conv3_blocks = []
        for idx in range(conv3_block_configs.get('n_blocks')):
            self.conv3_blocks.append(self.ResBlock(**conv3_block_configs.get('block_configs')))

        conv4_block_configs = res_blocks_configs.get('conv4_block_configs')
        self.conv4_blocks = []
        for idx in range(conv4_block_configs.get('n_blocks')):
            self.conv4_blocks.append(self.ResBlock(**conv4_block_configs.get('block_configs')))

        conv5_block_configs = res_blocks_configs.get('conv5_block_configs')
        self.conv5_blocks = []
        for idx in range(conv5_block_configs.get('n_blocks')):
            self.conv5_blocks.append(self.ResBlock(**conv5_block_configs.get('block_configs')))

        self.conv2d_3 = layers.Conv2D(**net_configs.get('conv2d_3'))

        self.global_avg_pool = layers.GlobalAveragePooling2D()

        self.dense_layer = layers.Dense(**net_configs.get('dense_layer'))

        self.dropout_layer = layers.Dropout(**net_configs.get('dropout_layer'))

        self.classifier = layers.Dense(net_configs.get('number_of_classes'))

    def call(self, inputs, training=False):
        x = self.conv2d_1(inputs, training=training)

        x = self.conv2d_2(x, training=training)

        x = self.max_pool2d(x)

        for conv2_block in self.conv2_blocks:
            x = conv2_block(x)

        for conv3_block in self.conv3_blocks:
            x = conv3_block(x)

        for conv4_block in self.conv4_blocks:
            x = conv4_block(x)

        for conv5_block in self.conv5_blocks:
            x = conv5_block(x)

        x = self.conv2d_3(x)

        x = self.global_avg_pool(x)

        x = self.dense_layer(x)

        x = self.dropout_layer(x)

        return self.classifier(x)

    def model(self):
        x = keras.Input(shape=self.input_image_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))
