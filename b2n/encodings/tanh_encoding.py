import tensorflow.keras as keras
import math
import numpy as np

def class_encoding(y_train, nb_classes=1):
    '''for tanh between 0 and 1.
        y_train must be an array
        nb_classes for the class count, if not all values in y_train
    '''
    
    return (y_train/5.0)-1

def class_decoding(y_train, nb_classes=1):
    ''' from tanh to 0.0-9.9 like function. It returns a float value between 0.0 and 9.9
        y_train the encoded values in an array
        nb_classes should be ignored. will not used
    '''
    
    return (y_train+1)*5.0