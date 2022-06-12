import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def tmnist_percentals(input_dir="datasets", max_count=28 ):
    y_train = np.empty((0))
    x_train = np.empty((0, 32, 20,3))
    
    # fonts in blacklist will be removed from dataset
    blacklist_url = input_dir + "/font-blacklist.txt"
    blacklist_data = pd.read_csv(blacklist_url, index_col=False)
    
    for i in range(max_count):

        dataset_url = input_dir + "/TMNIST_PERCENTAL_"+str((i+1)*1000)+"_Data.csv"
        data = pd.read_csv(dataset_url, index_col=False)
        data = data.drop(data[data['names'].isin(blacklist_data.values.reshape(-1))].index)
        
        y_tmnist = data[['labels']]
        
        X = data.drop({'labels','names'},axis=1)

        # resizing needs a new shape
        X_images = (X.values.reshape(-1,28,28,1))
        
        # resize but not all for padding
        X_images = tf.image.resize(X_images, (30,22))
        
        # now pad to make a border (later white)
        # so it looks more like a meter digit
        X_images = tf.image.resize_with_pad(X_images, 32, 20)
        
        # remove the shape for resizing
        X_images = np.array(X_images).reshape(-1,32,20)
        # gray to rgb (split in 3 channels)
        X_images = np.stack((X_images,)*3, axis=-1)
        
        x_train = np.concatenate((X_images, x_train))
        y_train = np.concatenate((y_tmnist.values.reshape(-1), y_train))
    
    return shuffle(x_train,  y_train, n_samples=len(y_train))