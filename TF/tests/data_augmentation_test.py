import os
import io
import datetime as dt
import numpy as np
import pathlib
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2


# Local imports
os.getcwd()
os.chdir('C:\\Users\\mchls\\Desktop\\Projects\\Deep-Learning\\TF')

from models.cnn import ConvModel
from callbacks.tensorboard_callbacks import ConvLayerVis
from aux_code import aux_funcs


DATA_PATH = Path('C:\\Users\\mchls\\Desktop\\Projects\\Data\\antrax')
BATCH_SIZE = 32
INPUT_IMAGE_SHAPE = (128, 128, 1)
CENTRAL_CROP_PROP = .7
CROP_SHAPE = INPUT_IMAGE_SHAPE
BRIGHTNESS_DELTA = 0.1
CONTRAST = (0.4, 0.6)


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



def figure_to_image(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def strip_layer_name(layer_full_name, layer_index=''):
    layer_stripped_name = layer_full_name + layer_index
    if find_sub_string(layer_full_name, '_'):
        if find_sub_string(layer_full_name, 'conv'):
            layer_full_name_bw = layer_full_name[::-1]
            layer_stripped_name = layer_full_name[:layer_full_name_bw.index('_')] + layer_index
    return layer_stripped_name

def find_sub_string(string: str, sub_string: str):
    return True if string.find(sub_string) > -1 else False

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
        output_layers = [layer.output for layer in self.model.model.layers if find_sub_string(layer.name, 'conv2d') or find_sub_string(layer.name, 'max_pooling2d')]

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
            fig, ax = plt.subplots(rows, cols, figsize=self.figure_configs.get('figsize'))
            for row in range(rows):
                for col in range(cols):
                    ax[row][col].imshow(feature_map[0, :, :, row+col], cmap=self.figure_configs.get('cmap'))

            plt.show()

            if isinstance(self.save_dir, pathlib.Path):
                if not self.save_dir.is_dir():
                    os.makedirs(self.save_dir)
                    fig.savefig(self.save_dir / f'./Epoch_{epoch}.png')
            with self.file_writer.as_default():
                tf.summary.image(f' feature map (epoch #{epoch})', figure_to_image(fig), step=epoch)

    def on_train_end(self, logs=None):
        pass


def preprocessing_func(image):
    img = tf.image.central_crop(image, CENTRAL_CROP_PROP)
    if img.shape[2] == 3:
        img = tf.image.rgb_to_grayscale(img)
    return img


def augment(image):
    img = tf.image.random_crop(image, size=CROP_SHAPE)  # Slices a shape size portion out of value at a uniformly chosen offset. Requires value.shape >= size.
    img = tf.image.random_brightness(img, max_delta=BRIGHTNESS_DELTA)  # Equivalent to adjust_brightness() using a delta randomly picked in the interval [-max_delta, max_delta)
    img = tf.image.random_contrast(img, lower=CONTRAST[0], upper=CONTRAST[1])  # Equivalent to adjust_contrast() but uses a contrast_factor randomly picked in the interval [lower, upper).
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return img


def load_image(image_file):
    # 1) Decode the path
    image_file = image_file.decode('utf-8')

    # 2) Read the image
    img = cv2.imread(image_file)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=-1)
    img = preprocessing_func(image=img)
    img = augment(img)
    img = tf.cast(img, tf.float32)
    img.set_shape([128, 128, 1])
    # 3) Get the label
    label = tf.strings.split(image_file, "\\")[-1]
    label = tf.strings.substr(label, pos=0, len=1)
    label = tf.strings.to_number(label, out_type=tf.float32)
    label = tf.cast(label, tf.float32)
    label.set_shape([])
    return img, label


def create_dataset():
    # 1) Rename the files to have consequent name
    idx = 1
    for root, folders, files in os.walk(DATA_PATH / 'train'):
        for file in files:
            os.rename(f'{root}/{file}', f'{root}/{idx}.tiff')
            idx += 1

    idx = 1
    for root, folders, files in os.walk(DATA_PATH / 'test'):
        for file in files:
            os.rename(f'{root}/{file}', f'{root}/{idx}.tiff')
            idx += 1

    # # 2) Group the images with their labels
    # idx = 1
    # for root, folders, files in os.walk(DATA_PATH / 'train'):
    #     for file in files:
    #         label = f'{file[:file.index(".")]}'
    #
    #         # 1) Create a new folder for the images
    #         label_dir = f'{root}/{label}'
    #         # os.mkdir(label_dir)
    #
    #         # 2) Move the original image to the new folder
    #         old_location = f'{root}/{file}'
    #         new_location = f'{label_dir}/{file}'
    #
    #         # 3) Load the image
    #         img, label = load_image(old_location)
    #         print(label)
    #         os.remove(old_location)
    #         for n in range(N_AUGS):
    #             # 1) Augment the image
    #             aug_img, _ = augment(image=img, label=label)
    #             # 2) To save the augmented image we need to reshape it from (w, h, c) to (w, h)
    #             aug_img = aug_img.numpy().reshape(aug_img.shape[:2])
    #             # 3) Conver numpy array to PIL Image
    #             aug_img = Image.fromarray(aug_img)
    #             # 4) Save the image
    #             aug_img.save(f'{label_dir}/{label}_{n}.tiff')


if __name__=='__main__':
    # create_dataset()

    train_ds = tf.data.Dataset.list_files(str(DATA_PATH / 'train/10,000x - 48/*.tiff'))
    train_ds = train_ds.map(lambda x: tf.numpy_function(load_image, [x], (tf.float32, tf.float32))).batch(BATCH_SIZE)
    # train_ds = train_ds.batch(BATCH_SIZE)
    # train_ds = train_ds.shuffle(buffer_size=1000)
    # train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    # train_ds = train_ds.repeat()

    X, y = next(iter(train_ds))
    X.shape.as_list(), y.shape.as_list()
    # Model with keras.Sequential
    model = ConvModel(input_shape=INPUT_IMAGE_SHAPE)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        metrics=['accuracy']
    )
    model.model.summary()
    train_log_dir = f'./logs/{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/train_data'
    train_file_writer = tf.summary.create_file_writer(train_log_dir)

    feat_maps_callback = ConvLayerVis(
        X=X,
        figure_configs=dict(rows=8, cols=8, figsize=(15, 15), cmap='gray'),
        file_writer=train_file_writer,
        save_dir = Path('./filters')
    )

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=train_log_dir,
            write_images=True
        ),
        feat_maps_callback
    ]

    model.fit(
        train_ds,
        # train_dg,
        batch_size=32,
        # steps_per_epoch=10,
        epochs=10,
        callbacks=callbacks
    )

    dg = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    zoom_range=(0.95, 0.95),
    horizontal_flip=True,
    vertical_flip=True,
    data_format='channels_last',
    validation_split=0.1,
    dtype=tf.float32
    )

    train_dg = dg.flow_from_directory(
    DATA_PATH / 'train',
    target_size=(128, 128),
    batch_size=32,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=True,
    subset='training'
    )
    img_file = 'C:\\Users\\mchls\\Desktop\\Projects\\Data\\antrax\\train\\10,000x - 48\\2.tiff'
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=-1)
    img.shape
    plt.imshow(img, cmap='gray')
    type(img)
    # img = np.array(Image.open(img_file))
    # img = np.expand_dims(img, axis=-1)
    # img.shape
    # plt.imshow(img)
    # train_ds = keras.preprocessing.image_dataset_from_directory(
    #     DATA_PATH  / 'train',
    #     labels='infered',
    #     label_mode='int',
    #     color_mode='grayscale',
    #     batch_size=BATCH_SIZE,
    #     image_size=IMAGE_SHAPE,
    #     shuffle=True,
    #     validation_split=0.1,
    #     subset='training'
    # )
    #
    # train_ds = train_ds.map(augment)
