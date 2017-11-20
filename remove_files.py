'''After creating class/file directories from 1_move_files.py, run this to
    move each file into the final resting class and directory. First move all
    videos associated to each class, then remove any directory/videos not to be
    trained on
'''

import os
import shutil



def rename_move_remove_files():
    classification_dict = { 'throwing': ['BaseballPitch','JavelinThrow'],
                            'punching': ['BoxingPunchingBag', 'BoxingSpeedBag'],
                            'jumping': ['HighJump', 'TrampolineJumping'],
                            'pull_ups': ['PullUps'],
                            'push_ups': ['PushUps']
                             }
    _set = ['train', 'test']

    for s in _set:
        for class_name in classification_dict.keys():
            #making a directory for each class
            if not os.path.exists(s+'/'+class_name):
                os.makedirs(s+'/'+class_name)

            for group in classification_dict[class_name]:
                #getting a list of files in each group
                filename = os.listdir(s + '/' + group)


                #moving files to correct directory
                for f in filename:
                    shutil.move(s+'/' + group +'/'+f, s + '/' + class_name + '/'+ f)


    #remove unecessary directories/files

def remove_files():
    classification_dict = { 'throwing': ['BaseballPitch','JavelinThrow'],
                            'punching': ['BoxingPunchingBag', 'BoxingSpeedBag'],
                            'jumping': ['HighJump', 'TrampolineJumping'],
                            'pull_ups': ['PullUps'],
                            'push_ups': ['PushUps']
                             }

    _set = ['train', 'test']
    for s in _set:
        for directory in os.listdir(s):
            if (directory not in classification_dict.keys() and directory != '.DS_Store'):
                shutil.rmtree(s + '/' + directory)

def main():
    rename_move_remove_files()

    remove_files()

if __name__ == '__main__':
    main()
