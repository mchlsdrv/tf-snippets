import os
import datetime as dt
import numpy as np
import pathlib
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, callbacks
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Lambda
from sklearn.neighbors import NearestNeighbors
import cv2
'''
You can adjust the verbosity of the logs which are being printed by TensorFlow
by changing the value of TF_CPP_MIN_LOG_LEVEL:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

    def save(self, save_path):
        self.model.save(save_path)


def strip_layer_name(layer_full_name, layer_index=''):
    layer_stripped_name = layer_full_name + layer_index
    if find_sub_string(layer_full_name, '_'):
        if find_sub_string(layer_full_name, 'conv'):
            layer_stripped_name = layer_full_name.split('_')[-1] + layer_index
    return layer_stripped_name


class ConvLayerVis(keras.callbacks.Callback):
    def __init__(self, X, figure_configs, file_writer, save_dir: pathlib.Path = None):
        super().__init__()
        self.X = X
        self.file_writer = file_writer
        self.figure_configs = figure_configs
        self.save_dir = save_dir

    @staticmethod
    def find_sub_string(string: str, sub_string: str):
        return True if string.find(sub_string) > -1 else False

    @staticmethod
    def figure_to_image(figure):
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image = tf.image.decode_png(buffer.get_value(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def on_training_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # 1) Get the layers
        output_layers = [layer for layer in self.model.model.layers if self.find_sub_string(layer.name, 'conv2d') or self.find_sub_string(layer.name, 'max_pooling2d')]

        # 3) Build partial model
        partial_model = keras.Model(
            inputs=self.model.model.input,
            outputs=output_layers
        )

        # 4) Get the feature maps
        feature_maps = partial_model.predict(self.X)

        # 5) Plot
        rows, cols = self.figure_configs.get('rows'), self.figure_configs.get('cols')
        for feature_map in feature_maps:
            fig, ax = plt.subplots(rows, cols, figsize=figure_configs.get('figsize'))
            for row in range(rows):
                for col in range(cols):
                    ax[row][col].imshow(feature_map[0, :, :, row+col], cmap=figure_configs.get('cmap'))

            plt.show()

            if isinstance(save_dir, pathlib.Path):
                if not save_dir.is_dir():
                    os.makedirs(save_dir)
                    fig.savefig(save_dir / f'./Epoch_{epoch}.png')

            with file_writer.as_default(f' feature map (epoch #{epoch})', figure_to_image(fig), step=epoch):
                tf.summary.image()

    def on_train_end(self, logs=None):
        pass


def get_patch_df(image_file, patch_height, patch_width):
    assert image_file.is_file(), f'No file \'{image_file}\' was found!'

    img = cv2.imread(str(image_file))
    df = pd.DataFrame(columns=['file', 'image'])
    img_h, img_w, _ = img.shape
    for h in range(0, img_h, patch_height):
        for w in range(0, img_w, patch_width):
            patch = img[h:h+patch_height, w:w+patch_width, :]
            df = df.append(dict(file=image_file, image=patch), ignore_index=True)
    return df


def transform_images(images_root_dir, model, patch_height, patch_width):
    df = pd.DataFrame(columns=['file', 'image'])
    for root, dirs, files in os.walk(images_root_dir):
        for file in files:
            df = df.append(get_patch_df(image_file=Path(f'{root}/{file}'), patch_height=patch_height, patch_width=patch_width), ignore_index=True)
    df.loc[:, 'vector'] = df.loc[:, 'image'].apply(lambda x: model(np.expand_dims(x, axis=0)) if len(x.shape) < 4 else model(x))
    return df

def get_knn_files(X, files, k):
    # Detect the k nearest neighbors
    nbrs_pred = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    nbrs_files = list()
    for idx, (file, x) in enumerate(zip(files, X)):
        _, nbrs_idxs = nbrs_pred.kneighbors(np.expand_dims(x, axis=0))
        nbrs_files.append(files[nbrs_idxs])
    return nbrs_files

if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    save_dir = Path('C:/Users/mchls/Desktop/Projects/Deep-Learning/TF/tests')
    images_dir = Path(('C:/Users/mchls/Desktop/Projects/Data/antrax/train/10,000x - 41'))
    # Load the data
    (X, y), (X_test, y_test) = cifar10.load_data()
    X, X_test = X.astype(np.float32) / 255.0, X_test.astype(np.float32) / 255.0
    w, h, c = X.shape[1], X.shape[2], X.shape[3]
    print(w, h, c)

    # Model with keras.Sequential
    model = ConvModel(input_shape=(w, h, c))

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        metrics=['accuracy']
    )

    train_log_dir = f'./logs/{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/train_data'
    train_file_writer = tf.summary.create_file_writer(train_log_dir)

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=train_log_dir,
            write_images=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_dir / 'checkpoints',
            verbose=1,
            save_weights_only=True,
            save_freq=5*32
        )
    ]

    model.fit(
        X,
        y,
        batch_size=64,
        epochs=10,
        callbacks=callbacks
    )
    model.model.save(save_path / 'model.h5')
    model_2 = tf.keras.models.load_model(save_path / 'model.h5')

    H = W = 32
    df = transform_images(images_root_dir=images_dir, model=model, patch_height=H, patch_width=W)
    df.shape
    df.loc[0, 'vector'].shape
    df.loc[0, 'image'].shape
    plt.imshow(df.loc[0, 'image'])
    df.to_pickle(save_dir / 'df.pkl')
    df_1 = pd.read_pickle(save_dir/'df.pkl')
    df_1
    LS = [model_2.predict(np.expand_dims(x, axis=0)) for x in X]
    LS = np.array(LS).reshape(50000, 10)
    LS.shape
    df.loc[:, 'vector'].values[0][0].numpy()

    # Detect the k nearest neighbors
    # X = np.array([x[0].numpy() for x in df.loc[:, 'vector'].values])
    # files = df.loc[:, 'file'].values
    # X.shape
    # nbrs_pred = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)
    # nbrs_pred
    # nbrs_files = list()
    # for idx, (file, x) in enumerate(zip(files, X)):
    #     _, idxs = nbrs_pred.kneighbors(np.expand_dims(x, axis=0))
    #     nbrs_files.append(files[idxs])
    # nbrs_files
    X = np.array([x[0].numpy() for x in df.loc[:, 'vector'].values])
    files = df.loc[:, 'file'].values
    df.loc[:, 'neighbors'] = get_knn_files(X=X, files=files, k=5)
    df
    plt.imshow(df.loc[2, 'image'])









