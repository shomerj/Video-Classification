from keras.models import Sequential
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.optimizers import SGD
from keras.layers import ZeroPadding3D, Dense, Flatten, Dropout

def get_model(input_shape, classes):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st layer group
    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                     padding='same', name='conv1',
                     strides=(1, 1, 1),
                     input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1,2,2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                     padding='same', name='conv2',
                     strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3a',
                     strides=(1, 1, 1)))
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3b',
                     strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2),
                           padding='valid', name='pool3'))
    # # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4a',
                     strides=(1, 1, 1)))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4b',
                     strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3.5'))

    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5a',
                     strides=(1, 1, 1)))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5b',
                     strides=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1))))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    model.add(Flatten())

    # FC layers group
    model.add(Dense(2024, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(2024, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    return model
