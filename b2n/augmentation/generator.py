import cv2
import numpy as np
import random
from scipy.stats import norm
import cv2
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def invert(imagem):
    if (random.getrandbits(1)):
        return (255)-imagem
    else:
        return imagem
        


def augmentation(x, y, augmentations={"Width_Shift_Range" : .2,
"Height_Shift_Range" : .1,
"Brightness_Range" : [0.4,0.9],
"Rotation_Angle": 3,
"ZoomRange_Out" : 0.1,
"ZoomRange_In" : 0.1,
"ShearRange" : 2,
"Channel_Shift" : 0.2,
"zca_whitening" : True
}, Batch_Size = 32):

    datagen = ImageDataGenerator(width_shift_range=augmentations["Width_Shift_Range"], 
                             height_shift_range=augmentations["Height_Shift_Range"],
                             brightness_range=augmentations["Brightness_Range"],
                             zoom_range=[1-augmentations["ZoomRange_In"], 1+augmentations["ZoomRange_Out"]],
                             rotation_range=augmentations["Rotation_Angle"],
                             zca_whitening=augmentations["zca_whitening"],
                             #channel_shift_range=Channel_Shift,
                             fill_mode='nearest',
                             shear_range=augmentations["ShearRange"]
                             ,preprocessing_function=invert
                             ,dtype=float
                             ,rescale=1./255)
    datagen.fit(x)                         
    return datagen.flow(x, y, batch_size=Batch_Size)

def augmentation_validation(x, y, Batch_Size = 32):

    datagen = ImageDataGenerator(preprocessing_function=invert
                             ,dtype=float
                             ,rescale=1./255)
    return datagen.flow(x, y, batch_size=Batch_Size)
