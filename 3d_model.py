from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD, Adam, RMSprop
from process_data_3d_generator import ProcessData
from cov3d import get_model
import ipdb


def train_model(seq_len, img_size):
    epoch = 10
    batch = 30
    input_shape = (30, 100,100, 1)
    #getting data for both test and train
    data_train = ProcessData(seq_len=seq_len, image_shape=img_size)
    train_generator = data_train.generate_images_in_memory('train', batch, avg=False)
    # ipdb.set_trace()
    # X = data_train.X
    # y = data_train.y

    data_test = ProcessData(seq_len=seq_len, image_shape=img_size)
    test_generator = data_test.generate_images_in_memory('test', batch, avg=False)

    # X_test = data_test.X
    # y_test = data_test.y
    # ipdb.set_trace()
    #loading the model
    # shape = data_train.input_shape
    labels = len(data_train.labels)
    model = get_model(input_shape, labels)

    metrics=['accuracy']
    steps = len(data_train.data)//batch

    optimizer = Adam(lr=1e-4, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

    model.fit_generator(
                generator=train_generator,
                steps_per_epoch,
                validation_data=test_generator,
                validation_steps=10
                verbose=1,
                epochs=epoch)

    print(model.summary())


def main():
    seq_len = 30
    image_size = (100,100)
    train_model(seq_len, image_size)

if __name__ == '__main__':
    main()
