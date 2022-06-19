def load_dataset(dataset_dir):
    
    ds = tf.keras.utils.image_dataset_from_directory(directory=dataset_dir, 
                                            image_size=(32,28), 
                                            batch_size=32,
                                            color_mode='rgb',
                                            label_mode='categorical',
                                            shuffle=True)
    return ds