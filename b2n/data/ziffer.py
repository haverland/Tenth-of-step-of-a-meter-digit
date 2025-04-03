import os
from PIL import Image 
import numpy as np


def ziffer_data_files(input_dir):
    imgfiles = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if (file.endswith(".jpg") and not file.startswith("10_") and not file.startswith("N")):
                #print(root + "/" + file)
                imgfiles.append(root + "/" + file)
    return  imgfiles

def ziffer_data(input_dir='images', input_shape=(32,20,3)):
    
    files = ziffer_data_files(input_dir)
    
    y_data = np.empty((len(files)))
    y_file = np.empty((len(files)), dtype="<U250")
    x_data = np.empty((len(files),input_shape[0],input_shape[1],input_shape[2]))

    for i, aktfile in enumerate(files):
        base = os.path.basename(aktfile)

        # get label from filename (1.2_ new or 1_ old),
        if (base[1]=="." and not base[2] == 'j'):
            target = base[0:3]
        else:
            target = base[0:1]
        #print(base)
        category = float(target)
        
        test_image = Image.open(aktfile).resize((input_shape[1],input_shape[0]))
        
        # convert to grayscale if input_shape = set to 1 for colorchannels
        if (input_shape[2] == 1):
            test_image = test_image.convert("L")
    
        test_image = np.array(test_image, dtype="float32")

        if (input_shape[2] == 1):
            test_image = np.expand_dims(test_image, axis=2)
        y_file[i] =  aktfile
        x_data[i] = test_image
        y_data[i] =  category
    print("Ziffer data count: ", len(y_data))   
    return x_data, y_data, y_file