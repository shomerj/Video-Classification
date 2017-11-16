import csv
import glob
import os
import os.path
from subprocess import call
import ipdb


def extract_images(fps=12):
    '''
    Once files are in there associated class folders we must extract frame for
    each video and place them in correct folder. Path to sequnce will look like:
    train/throwing/seq_1/<images.png>

    Extracting will be done with ffmpeg:
    `ffmpeg -i v_BoxingPunchingBag_g10_c02.avi -vf fps=10,scale=160:120 image%d.png`
    '''

    folders = ['test', 'train']
    data_file = []
    for folder in folders:
        class_folder = glob.glob(folder + '/*')
        # ipdb.set_trace()


        for vid_class in class_folder:
            class_filepath = glob.glob(vid_class + '/*')
            seq = 0
            for vid_path in class_filepath:
                test_train, label, path =  vid_path.split('/')

                if not os.path.exists(test_train + '/' + label + '/' + 'seq_' + str(seq)):
                    os.makedirs(test_train + '/' + label + '/' + 'seq_' + str(seq))

                src = test_train + "/" + label + '/' + path
                dest = test_train + "/" + label + '/' + 'seq_'+str(seq) +'/' + label+'_' + str(seq) +'-%03d.png'
                call(["ffmpeg", "-i", src, '-vf', 'fps='+str(fps) , dest])

                directory_path = test_train + "/" + label + '/' + 'seq_'+str(seq)
                nb_of_frames = get_frame_count(directory_path)

                data_file.append([test_train, label, seq, nb_of_frames, directory_path])
                seq+=1

    with open('image_file.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(data_file)


def get_frame_count(directory_path):
    #this will count the number of frames per video
    frame_count = glob.glob(directory_path+'/*')
    return len(frame_count)


def main():
    extract_images(fps=15)

if __name__ == '__main__':
    main()
