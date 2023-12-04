import tensorflow as tf
from config import *


def MobileNet():
    model = tf.keras.applications.mobilenet.MobileNet(include_top=True,
                                                      weights=None,
                                                      input_shape=(image_height, image_width, channels),
                                                      classes=5)
    return model