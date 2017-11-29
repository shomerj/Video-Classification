from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from skimage.transform import resize
from keras.layers import TimeDistributed, Dense, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from process_data_3d_generator import ProcessData
import numpy as np
from skimage import io
import matplotlib.pyplot as plt


def plot_image(img, name):
    fig = plt.figure(figsize=(12, 12))
    for i in range(img.shape[0]):
        ax = fig.add_subplot(5, 5, i+1)
        ax.imshow(img[i,:,:,:])
    plt.savefig('../images/'+name)
    plt.show()



def get_activations(model, layer, img):
    '''
    model: neural net model
    layer: the layer of your model you wish to visualize
    img: img to be visualized through activations
    '''
    activations_f = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    activations = activations_f((img, False))
    return activations

def plot_activation(n, img, name, model, layer_idx, frame):
    '''
    n: (int) nxn grid of images
    img: (numpy array)
    model
    layer_idx: (int) convolutional layer
    frame: (int) <26 frame in sequence
    '''
    n = n
    fig = plt.figure(figsize=(12, 12))
    img1 = np.expand_dims(img, axis=0)
    layer = model.layers[layer_idx]
    activations = get_activations(model, layer, img1)
    activation = activations[0]
    activated_img = activation[0][frame]

    for i in range(n):
        for j in range(n):
            idx = (n*i)+j
            ax = fig.add_subplot(n, n, idx+1)
            ax.imshow(activated_img[:,:,idx])
    plt.savefig('../images/'+name)
    plt.show()

def plot_activation_multi(n, image, model, frame, name):
    '''
    n: (int) nxn grid of images
    image: (numpy array) image after processed through convolutional layer
    model: keras model
    frame: (int) index of the frame in sequence
    '''
    layers = model.layers
    fig = plt.figure(figsize=(30, 30))
    img = np.expand_dims(image, axis=0)

    for i, layer in enumerate(layers):
        activations = get_activations(model, layer, img)
        activation = activations[0]
        activated_img = activation[0][frame]
        for j in range(1,n+1):
            idx = (n*i)+j
            ax = fig.add_subplot(len(layers), n, idx)
            ax.imshow(activated_img[:,:,j*3])
    plt.savefig('../images/'+name)
    plt.show()


def truncate_model(model):
    model_truncated = Sequential()
    model_truncated.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
        activation='relu', padding='same'), input_shape=(25,200,200,3)))
    model_truncated.add(TimeDistributed(Conv2D(32, (3,3),
        kernel_initializer="he_normal", activation='relu')))
    model_truncated.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model_truncated.add(TimeDistributed(Conv2D(64, (3,3),
        padding='same', activation='relu')))
    model_truncated.add(TimeDistributed(Conv2D(64, (3,3),
        padding='same', activation='relu')))
    model_truncated.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model_truncated.add(TimeDistributed(Conv2D(128, (3,3),
        padding='same', activation='relu')))
    model_truncated.add(TimeDistributed(Conv2D(128, (3,3),
        padding='same', activation='relu')))
    model_truncated.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model_truncated.add(TimeDistributed(Conv2D(256, (3,3),
        padding='same', activation='relu')))
    model_truncated.add(TimeDistributed(Conv2D(256, (3,3),
        padding='same', activation='relu')))
    model_truncated.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model_truncated.add(TimeDistributed(Conv2D(512, (3,3),
        padding='same', activation='relu')))
    model_truncated.add(TimeDistributed(Conv2D(512, (3,3),
        padding='same', activation='relu')))
    model_truncated.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    for i, layer in enumerate(model_truncated.layers):
        layer.set_weights(model.layers[i].get_weights())

    model_truncated.compile(loss='categorical_crossentropy', optimizer=SGD(),
                  metrics=['accuracy'])

    return model_truncated





if __name__ == '__main__':
    #lrcn model
    model_lrcn = load_model('logs7/checkpoing.hdf5')
    lrcn_data = ProcessData(seq_len=25, image_shape=(200,200))
    X_test, y_test = lrcn_data.generate_images_in_memory('test', avg=False, BW=False)
    model_truncated = truncate_model(model_lrcn)



    #c3d model
    # model_c3d = load_model('logs8/checkpoing.hdf5')
    # c3d_data = ProcessData(seq_len=15, image_shape=(100,100))
    # X_test1, y_test1 = c3d_data.generate_images_in_memory('test', avg=False, BW=True)

    # img1 = np.expand_dims(X_test1[4], axis=0)
    # layer1 = model_c3d.layers[0]
    # activations1 = get_activations(model_c3d, layer1, img1)
    # activation1 = activations1[0]
    # activated_img1 = activation1[0][10]
    # plot_activation(5, activated_img1)
