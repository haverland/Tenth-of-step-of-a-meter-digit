import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os

def tmnist_percentals(input_dir, use_grayscale=True):
    y_train = np.empty((0,1))
    if (use_grayscale):
        x_train = np.empty((0, 32, 20,1))
    else:
        x_train = np.empty((0, 32, 20,3))
    
    blacklist_url = os.path.join(input_dir, "font-blacklist.txt")
    blacklist_data = pd.read_csv(blacklist_url, index_col=False)
    #print(blacklist_data.values.reshape(-1))
    
    for i in range(28):

        dataset_url = os.path.join(input_dir, "TMNIST_PERCENTAL_"+str((i+1)*1000)+"_Data.csv")
        data = pd.read_csv(dataset_url, index_col=False)
        data = data.drop(data[data['names'].isin(blacklist_data.values.reshape(-1))].index)
        #print(data.head())
        y_tmnist = data[['labels']]
        #print(y_tmnist[y_tmnist.labels>10])
        X = data.drop({'labels','names'},axis=1)

        #print(data)
        X_images = (X.values.reshape(-1,28,28,1))
        # resize but not all for padding
        X_images = tf.image.resize(X_images, (30,22))
        # now pad to make a border (later white)
        # so it looks more like a meter digit
        X_images = tf.image.resize_with_pad(X_images, 32, 20)
        #X_images = tf.image.resize(X_images, (32,20))
        X_images = 255-np.array(X_images).reshape(-1,32,20,1)
        X_images = X_images/255.
            
        if (use_grayscale!=True):
            X_images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_images))
            

        x_train = np.concatenate((X_images, x_train))
        y_train = np.concatenate((y_tmnist.values, y_train))
    #print(">10", y_train[y_train>10])
    return shuffle(x_train,  y_train, n_samples=len(y_train))
