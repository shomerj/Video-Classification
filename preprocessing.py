from skimage import io
from skimage.transform import resize
import ipdb
import numpy as np


def image_processing(img, size, as_BW=True):
    '''
    img: image path to be processesd
    size: tuple equal to desired size
    as_BW: True is picture is black and white
    '''
    # ipdb.set_trace()
    img = io.imread(img, as_grey=as_BW)
    resized = resize(img, size, mode='constant')
    return resized.reshape(100,100,1).astype(np.float32)
