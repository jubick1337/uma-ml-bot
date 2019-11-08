import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from tensorflow.python.keras.backend import set_session


class Predictor:
    def __create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=(224, 224, 3)))
        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(GlobalAveragePooling2D())

        model.add(Dense(256, activation='relu'))

        model.add(BatchNormalization())

        model.add(Dense(25, activation='softmax'))
        return model

    def predict(self, path):
        K.clear_session()
        img = img_to_array(load_img(path, target_size=(224, 224))) / 255
        img = img.reshape([-1, 224, 224, 3])
        global sess
        global graph
        with self.graph.as_default():
            set_session(self.sess)
            p = self.model.predict(img)
            return np.argmax(p)

    def __init__(self):
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.sess)
        self.model = self.__create_model()
        self.model.load_weights('model.h5')
