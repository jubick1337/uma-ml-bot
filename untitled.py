from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_crossentropy

my_model= models.Sequential()

# Add first convolutional block
my_model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                    input_shape=(224,224,3)))
my_model.add(MaxPooling2D((2, 2), padding='same'))

# second block
my_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))
# third block
my_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))
# fourth block
my_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))

# global average pooling
my_model.add(GlobalAveragePooling2D())
# fully connected layer
my_model.add(Dense(256, activation='relu'))
my_model.add(BatchNormalization())
# make predictions
my_model.add(Dense(25, activation='softmax'))

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc= ModelCheckpoint('/your_model.h5', monitor='val_loss',
                    mode='min', verbose=1, save_best_only=True)

my_model.compile(optimizer='adam', loss='categorical_crossentropy',
                 metrics=['accuracy'])

import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    print(token)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
        print(response)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

download_file_from_google_drive('1c3pTq4lDlrOY3QTIvGx6fnnsBccA-qwJ', 'uma.zip')
!unzip 'uma.zip'

data_generator = ImageDataGenerator(rescale=1./255)

train_generator = data_generator.flow_from_directory(
        '/content/images_train',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical',
        classes=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
        )

# get batches of validation images from the directory
validation_generator = data_generator.flow_from_directory(
        '/content/images_test',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical',
        classes=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
        )

train_generator.class_indices

history = my_model.fit_generator(
        train_generator,
        epochs=100,
        validation_data=validation_generator)

x, y = next(train_generator)

import matplotlib.pyplot as plt
import numpy as np
x, y = next(validation_generator)
for i in range (0,3):
    image = x[i]
    print(np.argmax(y[i]))
    print(np.argmax(my_model.predict(image.reshape([-1,224,224,3]))))
    plt.imshow(image)
    plt.show()

my_model.save('model_n.h5')

from keras.utils import plot_model
plot_model(my_model, to_file='model.png')

my_model.evaluate_generator(validation_generator)
