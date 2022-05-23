import cv2
import numpy as np
import random
from scipy.stats import norm
import cv2
import random
from keras.preprocessing.image import ImageDataGenerator



def invert(imagem):
    if (random.getrandbits(1)):
        return (255)-imagem
    else:
        return imagem
        

Shift_Range = 1 # px
Brightness_Range = [0.4,0.9]
Rotation_Angle = 3
ZoomRange_Out = 0.1
ZoomRange_In = 0.1
ShearRange= 2
Channel_Shift=0.2


def augmentation(x, y, Batch_Size = 32):
    datagen = ImageDataGenerator(width_shift_range=Shift_Range, 
                             height_shift_range=Shift_Range,
                             brightness_range=Brightness_Range,
                             zoom_range=[1-ZoomRange_In, 1+ZoomRange_Out],
                             rotation_range=Rotation_Angle,
                             #channel_shift_range=Channel_Shift,
                             fill_mode='nearest',
                             shear_range=ShearRange
                             ,preprocessing_function=invert
                             ,dtype=float
                             ,rescale=1./255)
    return datagen.flow(x, y, batch_size=Batch_Size)
