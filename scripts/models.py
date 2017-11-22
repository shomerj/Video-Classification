from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Conv3D, MaxPooling3D


class Models():

    def __init__(self, model, label,  seq_len):
        self.classes = label
        self.seq_len = seq_len

        if model == "cnn_lstm":
            self.input_shape = (self.seq_len, 227,227,1)
            self.model = self.cnn_lstm()
        elif model == 'c3d':
            self.input_shape = (self.seq_len, 100,100,1)
            self.model = self.c3d()

    def cnn_lstm(self):
        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
            activation='relu', padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3,3),
            kernel_initializer="he_normal", activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        #
        # model.add(TimeDistributed(Conv2D(512, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(512, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Flatten()))

        model.add(Dropout(0.5))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(128))
        model.add(Dense(self.classes, activation='softmax'))

        return self.model


    def c3d(self):
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
        model.add(Dense(self.classes, activation='softmax'))

        return self.model
