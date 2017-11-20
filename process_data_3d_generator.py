import pandas as pd
import numpy as np
import glob
import os
from keras.utils import np_utils
from preprocessing import image_processing
import ipdb
import threading



class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class ProcessData():

    def __init__(self, seq_len=16, image_shape=(100,100)):
        '''
        seq_len: the max length of a sequence of images in consideration
        image_shape: the target scaled image
        '''
        self.seq_len = seq_len
        self.image_shape = image_shape
        self.max_frames = 312
        self.squence_path = 'sequence/'
        self.data = self.load_data()
        self.labels = self.get_labels()
        self.X = None
        self.y = None
        self.average = None
        self.input_shape = None

    def load_data(self):
        '''
        columns: train/test, label, sequence, nb_frames, dir_path
        '''
        df = pd.read_csv('sample_data.csv')
        self.data = df
        return self.data


    def get_labels(self):
        self.labels = self.data['label'].unique()
        return self.labels


    def one_hot_encode_label(self, label_str):
        '''
        Encode label as index in self.labels.
        Then one hot encode encoded label_str
        '''
        label_list = self.labels.tolist()
        encoded = label_list.index(label_str)

        one_hot = np_utils.to_categorical(encoded, len(self.labels))
        # one_hot = one_hot

        return one_hot

    @threadsafe_generator
    def generator_images(self, train_test, batch_size, avg=True):
        '''
        Grabs images from disk and loads them into memory.
        train/test = str of test or train

        return: X = np.array of list of sequence of images
                y = np.array of labels
        '''
        test, train = self.train_test_split()

        #specifiying train or test set
        if train_test == 'train':
            data = train
        else:
            data = test
        indices = np.arange(len(data))
        np.random.shuffle(indices)

        while 1:
            X, y, average = [], [], []
            # indices = np.arange(len(data))
            # np.random.shuffle(indices)
            # ipdb.set_trace()
            for row in data.values[indices[:batch_size]]:

                frames = self.grab_frame_sequence(row)

                if len(frames) <= self.seq_len:
                    continue


                frames = self._create_sequence(frames, self.seq_len)
                sequence = self.build_seq_with_processing(frames, self.image_shape, BW=True)

                X.append(sequence)
                y.append(self.one_hot_encode_label(row[1]))
                indices = indices[batch_size:]

            self.average = np.array(average)
            X = np.array(X)
            yield X, np.array(y)

    def generate_images_in_memory(self, train_test, avg=False):
        '''
     Grabs images from disk and loads them into memory.
     train/test = str of test or train

     return: X = np.array of list of sequence of images
           y = np.array of labels
         '''

        test, train = self.train_test_split()

         #specifiying train or test set
        if train_test == 'train':
            data = train
        else:
            data = test

        X, y, average = [], [], []
        for row in data.values:

            frames = self.grab_frame_sequence(row)

            if len(frames) <= self.seq_len:
                continue


            frames = self._create_sequence(frames, self.seq_len)
            sequence = self.build_seq_with_processing(frames, self.image_shape, BW=True)

            X.append(sequence)
            y.append(self.one_hot_encode_label(row[1]))

        if avg == True:
            average = np.array(average)
            average = np.mean(average, axis=3)


        return (np.array(X), np.array(y), np.array(average))

    def train_test_split(self):
        '''returns two dataframes: test and train
        '''
        test_data = self.data[self.data['train_test']=='test']
        train_data = self.data[self.data['train_test']=='train']
        return test_data, train_data

    def grab_frame_sequence(self, row):
        '''
        grabbing all frames from a given sequence.
        sample: a row from train/test dataframe
        '''
        path = row[4]
        frames = glob.glob(path+'/*')
        return frames


    def _create_sequence(self, frames, length):
        '''
        create a sequence of frames roughly equal to length.
        return: list of frames
        '''

        skip = len(frames)//length
        frame_seq = [frames[i] for i in range(0, len(frames), skip)]
        return frame_seq[:length]



    def build_seq_with_processing(self, frames, shape, BW):
        '''Frames is a list of paths to the frames of a given video'''

        return [image_processing(img, self.image_shape, as_BW=BW) for img in frames]
