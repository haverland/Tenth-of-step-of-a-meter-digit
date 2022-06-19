import os
import shutil
from PIL import Image 
from tensorflow import keras
import numpy as np


def ziffer_data_files(input_dir):
    imgfiles = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if (file.endswith(".jpg") and not file.startswith("10_") and not file.startswith("N")):
                #print(root + "/" + file)
                imgfiles.append(root + "/" + file)
    return  imgfiles

def ziffer_data(input_dir='images'):
    
    files = ziffer_data_files(input_dir)
    
    y_data = np.empty((len(files)))
    y_file = np.empty((len(files)), dtype="<U50")
    x_data = np.empty((len(files),32,20,3))

    for i, aktfile in enumerate(files):
        base = os.path.basename(aktfile)

        # get label from filename (1.2_ new or 1_ old),
        if (base[1]=="."):
            target = base[0:3]
        else:
            target = base[0:1]
         
        category = float(target)
        
        test_image = Image.open(aktfile).resize((20, 32))
        test_image = np.array(test_image, dtype="float32")
        y_file[i] =  aktfile
        x_data[i] = test_image
        y_data[i] =  category
    print("Ziffer data count: ", len(y_data))   
    return x_data, y_data, y_file