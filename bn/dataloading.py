import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def tmnist_percentals():
    x_train = np.empty((1,32,20,3))
    y_train = np.empty((1,1))
    for i in range(28):

        dataset_url = "datasets/TMNIST_PERCENTAL_"+str((i+1)*1000)+"_Data.csv"
        data = pd.read_csv(dataset_url, index_col=False)
        #print(data.head())
        y_tmnist = data[['labels']]
        X = data.drop({'labels','names'},axis=1)

        #print(data)
        X_images = (255-(X.values.reshape(-1,28,28,1)))/255.
        x_tmnist = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_images))
        x_tmnist = tf.image.resize(x_tmnist, (32,20))

        x_train = np.concatenate((x_tmnist, x_train))
        y_train = np.concatenate((y_tmnist.values, y_train))
        #print(x_tmnist.shape, np.array(y_tmnist).shape)
    return shuffle(x_train,  y_train)
