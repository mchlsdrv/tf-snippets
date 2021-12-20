import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    img = np.random.randint(0, 255, 100*100).reshape((100, 100, 1))
    plt.imshow(img, cmap='gray')
    img_2 = np.expand_dims(img, axis=-1)
    img_2.shape
    plt.imshow(img_2)

    img = cv2.imread('C:/Users/mchls/Desktop/Projects/Deep-Learning/TF/tests/1.tiff')
    img.shape
    tf.image.crop_and_resize(img, boxes=[10, 4], crop_size=[128, 128])
    model = resnet.ResNet50(weights='imagenet', )