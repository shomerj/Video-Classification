from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Conv3D, MaxPooling3D
from keras.models import Sequential
from keras.optimizers import Adam


def cnn3d(input_shape, nb_classes):
    model = Sequential()
    model.add(Conv3D(64, (7,7,7),padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D((11,11,11), strides=(4,4,4)))
    # model.add(Conv3D(128, (3,3,3),padding='same', activation='relu'))
    # model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))
    # model.add(Conv3D(128, (3,3,3), padding='same', activation='relu'))
    # model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))
    # model.add(Conv3D(256, (3,3,3), padding='same',activation='relu'))
    # model.add(Conv3D(256, (3,3,3),padding='same',  activation='relu'))
    model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model
