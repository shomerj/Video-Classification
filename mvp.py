from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, MaxPooling2d
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from process_data import ProcessData


data = ProcessData()
