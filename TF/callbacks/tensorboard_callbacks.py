import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from aux_code import aux_funcs


class ConvLayerVis(keras.callbacks.Callback):
    def __init__(self, X, input_layer, layers, figure_configs: dict, log_dir: str, log_interval: int):
        super().__init__()
        self.X_test = X
        self.input_layer = input_layer
        self.layers = layers
        n_dims = len(self.X_test.shape)
        assert 2 < n_dims < 5, f'The shape of the test image should be less than 5 and grater than 2, but current shape is {self.X_test.shape}'

        # In case the image is not represented as a tensor - add a dimension to the left for the batch
        if len(self.X_test.shape) < 4:
            self.X_test = np.reshape(self.X_test, (1,) + self.X_test.shape)

        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.figure_configs = figure_configs
        self.log_interval = log_interval

    def on_training_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # 1) Get the layers
        if epoch % self.log_interval == 0:
            # 1) Get the layers
            output_layer_tuples = [(idx, layer) for idx, layer in enumerate(self.layers) if aux_funcs.find_sub_string(layer.name, 'conv2d') or aux_funcs.find_sub_string(layer.name, 'max_pooling2d')]
            # output_layer_tuples = [(idx, layer) for idx, layer in enumerate(self.model.model.layers) if find_sub_string(layer.name, 'conv2d') or find_sub_string(layer.name, 'max_pooling2d')]
            output_layers = [layer_tuple[1].output for layer_tuple in output_layer_tuples]

            # 2) Get the layer names
            conv_layer_name_tuples = [(layer_tuple[0], f'Layer #{layer_tuple[0]} - Conv 2D ') for layer_tuple in output_layer_tuples if aux_funcs.find_sub_string(layer_tuple[1].name, 'conv2d')]
            max_pool_layer_name_tuples = [(layer_tuple[0], f'Layer #{layer_tuple[0]} - Max Pooling 2D') for layer_tuple in output_layer_tuples if aux_funcs.find_sub_string(layer_tuple[1].name, 'max_pooling2d')]

            layer_name_tuples = (conv_layer_name_tuples + max_pool_layer_name_tuples)
            layer_name_tuples.sort(key=lambda x: x[0])

            layer_names = [layer_name_tuple[1] for layer_name_tuple in layer_name_tuples]

            # 3) Build partial model
            partial_model = keras.Model(
                inputs=self.input_layer,
                # inputs=model.model.input,
                outputs=output_layers
            )

            # 4) Get the feature maps
            feature_maps = partial_model.predict(self.X_test)

            # 5) Plot
            rows, cols = self.figure_configs.get('rows'), self.figure_configs.get('cols')
            for feature_map, layer_name in zip(feature_maps, layer_names):
                fig, ax = plt.subplots(rows, cols, figsize=self.figure_configs.get('figsize'))
                for row in range(rows):
                    for col in range(cols):
                        ax[row][col].imshow(feature_map[0, :, :, row+col], cmap=self.figure_configs.get('cmap'))
                fig.suptitle(f'{layer_name}')

                with self.file_writer.as_default():
                    tf.summary.image(f'{layer_name} Feature Maps', aux_funcs.get_image_from_figure(figure=fig), step=epoch)
