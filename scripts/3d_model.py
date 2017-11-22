from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from models import Models
from process_data_3d_generator import ProcessData
import ipdb


def train_model(model, seq_len, img_size, avg=True, generator=False):
    '''
    This script runs the model.
    model: (str) model name
    seq_len: (int) the number of frames per sequence
    img_size: (tuple) size you want to scale image to
    avg: (bool) if you want to subtract the mean from each images
    generator: (bool) if you want to load all images into memory

    '''
    epoch = 10000
    batch = 32



    #getting data for both test and train
    if generator==True:
        data_train = ProcessData(seq_len=seq_len, image_shape=img_size)
        train_generator = data_train.generator_images('train', batch, avg=False)

        data_test = ProcessData(seq_len=seq_len, image_shape=img_size)
        test_generator = data_test.generator_images('test', batch, avg=False)
    else:
        data_train = ProcessData(seq_len=seq_len, image_shape=img_size)
        X, y= data_train.generate_images_in_memory('train', avg=avg)

        data_test = ProcessData(seq_len=seq_len, image_shape=img_size)
        X_test, y_test= data_test.generate_images_in_memory('test', avg=avg)

    #loading the model
    labels = len(data_train.labels)
    md = Models(model, labels, seq_len)

    metrics=['accuracy']

    optimizer = Adam(lr=.0001, decay=5e-6)
    md.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)


    #figure out steps per epoch. 0.6 is roughly the amount of train data
    steps = (len(data_train.data))//batch

    #callbacks
    earlystopping =  EarlyStopping(monitor='val_loss', patience=5)
    modelcheckpoint = ModelCheckpoint(filepath='../logs/checkpoing.hdf5', verbose=1, save_best_only=True, period=4)
    csvlog = CSVLogger('../logs/training.log')
    tensorboard = TensorBoard(log_dir='../logs/', histogram_freq=2)

    if generator == True:
        md.model.fit_generator(
                    generator=train_generator,
                    steps_per_epoch=steps,
                    validation_data=test_generator,
                    validation_steps=40,
                    callbacks = [earlystopping, modelcheckpoint, csvlog, tensorboard],
                    verbose=1,
                    epochs=epoch)
    else:
        md.model.fit(
            X,
            y,
            batch_size=batch,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[earlystopping, modelcheckpoint, csvlog, tensorboard],
            epochs=epoch)

    print(model.summary())
    return md.model, X_test, y_test

def scores(model, X_test, y_test):
    y_pred = model.predict_classes(X_test)
    y_true = []
    for row in y_test:
        y_true.append(row.argmax())
    y_true = np.array(y_true)
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


def main():
    '''
    model =[c3d, cnn_lstm]
    '''
    seq_len = 25
    image_size = (100,100)
    model, X_test, y_test = train_model('c3d', seq_len, image_size)
    scores(model,X_test, y_test)


if __name__ == '__main__':
    main()
