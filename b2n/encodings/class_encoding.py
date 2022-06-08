from tensorflow import keras
import numpy as np

def class_encoding(y_train, nb_classes):
    '''like to_categorical.
        y_train must be an array
        nb_classes for the class count, if not all values in y_train
    '''
    ret = np.zeros((len(y_train), nb_classes))
    for i, y in enumerate(y_train):
        ret[i, int((y*10))] = 1
    return ret

def class_decoding(y_train, nb_classes=100):
    ''' from_categorical like function. It returns a float value between 0.0 and 9.9
        y_train the encoded values in an array
        nb_classes should be ignored. will not used
    '''
    ret = np.zeros((len(y_train), 1))
    for i, y in enumerate(y_train):
        ret[i] = (np.argmax(y))/10
    return ret