import tensorflow.keras as keras
import math
import numpy as np

def class_encoding(y_train, nb_classes):
    ret = np.zeros((len(y_train), nb_classes))
    print(ret.shape)
    for i, y in enumerate(y_train):
        ret[i, int(y)] = (1-(y-int(y)))
        ret[i, int((y+1)%10)] = y-int(y)
    return ret

def sin_cos_encoding(y_data):
    y_train = []

    for target_number in y_data:
        target_number = (target_number) / 10
        target_sin = math.sin(target_number * math.pi * 2)
        target_cos = math.cos(target_number * math.pi * 2)

        zw = np.array([target_sin, target_cos])
        y_train.append(zw)
        #print(target_number*10, round(sin_cos_encoding_revert(zw)*10,1))
    return np.array(y_train)

def sin_cos_encoding_revert(y):
    return (np.arctan2(y[0], y[1])/(2*math.pi)) % 1