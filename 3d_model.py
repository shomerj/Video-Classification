from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from keras.utils import plot_model
import pydot
import graphviz
import h5py
from cov3d import get_model
from process_data_3d_generator import ProcessData
from cn3d import cnn3d
import ipdb


def train_model(seq_len, img_size, generator=True):
    # ipdb.set_trace()
    epoch = 10000
    batch = 30
    input_shape = (seq_len, 100, 100, 1)


    #getting data for both test and train
    if generator==True:
        data_train = ProcessData(seq_len=seq_len, image_shape=img_size)
        train_generator = data_train.generator_images('train', batch, avg=False)

        data_test = ProcessData(seq_len=seq_len, image_shape=img_size)
        test_generator = data_test.generator_images('test', batch, avg=False)
    else:
        data_train = ProcessData(seq_len=seq_len, image_shape=img_size)
        X, y, avg = data_train.generate_images_in_memory('train')

        data_test = ProcessData(seq_len=seq_len, image_shape=img_size)
        X_test, y_test, avg_test = data_test.generate_images_in_memory('test')
    # ipdb.set_trace()

    #loading the model
    labels = len(data_train.labels)
    model = get_model(input_shape, labels)

    metrics=['accuracy']

    optimizer = Adam(lr=1e-4, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

    #figure out steps per epoch. 0.6 is roughly the amount of train data
    steps = (len(data_train.data)*0.6)//batch

    #callbacks
    earlystopping =  EarlyStopping(monitor='val_loss', patience=5)
    modelcheckpoint = ModelCheckpoint(filepath='logs/checkpoing.hdf5', verbose=1, save_best_only=True, period=4)
    csvlog = CSVLogger('logs/training.log')
    tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0)

    if generator == True:
        model.fit_generator(
                    generator=train_generator,
                    steps_per_epoch=steps,
                    validation_data=test_generator,
                    validation_steps=40,
                    callbacks = [earlystopping, modelcheckpoint, csvlog, tensorboard],
                    verbose=1,
                    epochs=epoch)
    else:
        model.fit(
            X,
            y,
            batch_size=batch,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tensorboard, earlystopping, csvlog],
            epochs=epoch)

    # print(model.summary())
    # score = model.evaluate(X_test, y_test, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

def main():
    seq_len = 20
    image_size = (100,100)
    train_model(seq_len, image_size)

if __name__ == '__main__':
    main()
