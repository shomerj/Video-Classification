from skimage import io
from skimage.transform import resize
# import ipdb
import numpy as np


def image_processing(img, size, as_BW=True):
    '''
    img: image path to be processesd
    size: tuple equal to desired size
    as_BW: True is picture is black and white
    '''
    # ipdb.set_trace()
    if as_BW ==True:
        channel =1
    else:
        channel=3
    img = io.imread(img, as_grey=as_BW)
    resized = resize(img, size, mode='constant')
    return resized.reshape(size[0], size[1], channel).astype(np.float32)
