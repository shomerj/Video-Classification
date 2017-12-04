import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.io import imread
from skimage.transform import resize


def load_data():
    '''
    columns: train/test, label, sequence, nb_frames, dir_path
    '''
    df = pd.read_csv('../image_data.csv')
    data = df
    return data


def create_frame_sequence(row, length):

    path = row[4]
    frames = glob.glob('../'+row+'/*')
    skip = len(frames)//length
    frame_seq = [frames[i] for i in range(0, len(frames), skip)]
    return frame_seq[-length:]


def plot_images(frames):
    n = len(frames)
    fig = plt.figure(figsize=(10, 10))
    for idx, img in enumerate(frames):
        for j in range(n):
            img1 = imread(img)
            ax = fig.add_subplot(5, 5, idx+1)
            ax.imshow(img1)

    # plt.tight_layout()
    plt.show()

def image_processing(img, size, as_BW=False):
    '''
    img: image path to be processesd
    size: tuple equal to desired size
    as_BW: True is picture is black and white
    '''
    # ipdb.set_trace()
    if as_BW ==True:
        channel=1
    else:
        channel=3
    img = io.imread(img, as_grey=as_BW)
    resized = resize(img, size, mode='constant')
    return resized.reshape(size[0], size[1], channel).astype(np.float32)

def plot_classification(recall, precision, f1):
    n = np.arange(5)
    width = .2
    fig, ax = plt.subplots()
    rec = ax.bar(n, recall, width, label='Recall', color='dodgerblue', edgecolor='black', alpha=.4)
    prec = ax.bar(n+width, precision, width, label='Precision', color='forestgreen',edgecolor='black',alpha=.4)
    f_1 = ax.bar(n+2*width, f1, width, label='F1', color='firebrick', edgecolor='black',alpha=.4)
    ax.set_ylabel('Scores')
    ax.set_title('Recall, Precision, F1 Scores by Class')
    ax.set_xticks(n + width / 2)
    ax.set_xticklabels(('Jumping', 'Pull Ups', 'Punching', 'Push Ups', 'Throwing'))
    ax.legend(loc=4)
    plt.savefig('../images/score_metrics')
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('../image_data.csv')

    #for each label
    df_throwing = df[df.label == 'throwing']
    df_punching = df[df.label == 'punching']
    df_jumping = df[df.label == 'jumping']
    df_pull_ups = df[df.label == 'pull_ups']
    df_push_ups = df[df.label == 'push_ups']

    #path to each video
    df_throwing.dir_path.values
    df_punching.dir_path.values
    df_jumping.dir_path.values
    df_pull_ups.dir_path.values
    df_push_ups.dir_path.values

    frame = create_frame_sequence(df_punching.dir_path.values[34], 25)
    plot_images(frame)

    recall = np.array([ 0.75362319,  0.14285714,  0.86046512,  0.13333333,  0.83783784])
    precision = np.array([ 0.88135593,  0.4       ,  0.7047619 ,  0.21052632,  0.65957447])
    f1 = np.array([ 0.8125    ,  0.21052632,  0.77486911,  0.16326531,  0.73809524])
    Accuracy =  0.6829268292682927

    plot_classification(recall, precision, f1)
