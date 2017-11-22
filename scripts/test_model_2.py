from keras.models import Sequential
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.optimizers import SGD
from keras.layers import ZeroPadding3D, Dense, Flatten, Dropout

def c3d(input_shape, classes):
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
                     strides=(2, 2, 2)))

    model.add(MaxPooling3D(pool_size=(4, 4, 4), strides=(2,2,2),
                           padding='valid', name='pool3'))
    model.add(Flatten())

    # FC layers group
    model.add(Dense((2048), activation='relu', name='fc4'))
    model.add(Dropout(0.2))
    model.add(Dense((1024), activation='relu', name='fc5'))
    model.add(Dropout(0.2))
    model.add(Dense((512), activation='relu', name='fc6'))
    model.add(Dropout(0.2))
    model.add(Dense(classes, activation='softmax'))

    return model
