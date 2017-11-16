from skimage import io
from skimage.transform import resize


def image_processing(img, size, as_BW=True):
    '''
    img: image path to be processesd
    size: tuple equal to desired size
    as_BW: True is picture is black and white
    '''
    img = io.imread(img, as_grey=as_BW)
    resized = resize(img, size)
    return resized
