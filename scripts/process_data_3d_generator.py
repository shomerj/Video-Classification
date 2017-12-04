import pandas as pd
import numpy as np
import glob
import os
from keras.utils import np_utils
from preprocessing import image_processing
import threading


class ProcessData():

    def __init__(self, seq_len=25, image_shape=(227,227)):
        '''
        seq_len: the max length of a sequence of images in consideration
        image_shape: the target scaled image
        '''
        self.seq_len = seq_len
        self.image_shape = image_shape
        self.squence_path = 'sequence/'
        self.data = self.load_data()
        self.labels = self.get_labels()

    def load_data(self):
        '''
        columns: train/test, label, sequence, nb_frames, dir_path
        '''
        df = pd.read_csv('../image_data.csv')
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

        return one_hot



    def generate_images_in_memory(self, train_test, avg=False, BW=False):
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

        X, y = [], []
        for row in data.values:
            frames = self.grab_frame_sequence(row)

            if len(frames) <= self.seq_len:
                continue


            frames = self._create_sequence(frames, self.seq_len)
            sequence = self.build_seq_with_processing(frames, self.image_shape, BW=BW)

            if avg == True:
                sequence = np.array(sequence)
                average = np.mean(sequence, axis=0)
                X.append(average-sequence)
                y.append(self.one_hot_encode_label(row[1]))

            else:
                X.append(sequence)
                y.append(self.one_hot_encode_label(row[1]))

        return np.array(X), np.array(y)

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
        frames = glob.glob('../'+path+'/*')
        return frames


    def _create_sequence(self, frames, length):
        '''
        create a sequence of frames roughly equal to length.
        return: list of frames
        '''

        skip = len(frames)//length
        frame_seq = [frames[i] for i in range(0, len(frames), skip)]
        return frame_seq[-length:]



    def build_seq_with_processing(self, frames, shape, BW):
        '''Frames is a list of paths to the frames of a given video'''

        return [image_processing(img, self.image_shape, as_BW=BW) for img in frames]
