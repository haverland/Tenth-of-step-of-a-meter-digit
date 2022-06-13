import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from b2n.encodings.class_encoding import class_decoding
import tensorflow as tf
from PIL import Image 
import os

def plot_dataset(images, labels, columns=10, rows=5, figsize=(18, 10)):

    fig = plt.figure(figsize=figsize)
    
    for i in range(1, columns*rows +1):
        if (i>len(labels)):
            break
        fig.add_subplot(rows, columns, i)
        plt.title(labels[i-1])  # set title
        plt.xticks([0.2, 0.4, 0.6, 0.8])
        plt.imshow((images[i-1]).astype(np.uint8), aspect='1.6', extent=[0, 1, 0, 1])
        # yellow lines
        for y in np.arange(0.2, 0.8, 0.2):
            plt.axhline(y=y,color='yellow')
        ax=plt.gca()
        ax.get_xaxis().set_visible(False) 
        plt.tight_layout()
    plt.show()

def plot_dataset_it(data_iter, columns=9, rows=5, nb_classes=100):

    fig = plt.figure(figsize=(18, 11))
    
    for i in range(1, columns*rows +1):
        img, label = data_iter.next()
        fig.add_subplot(rows, columns, i)
        plt.xticks([0.2, 0.4, 0.6, 0.8])
        plt.title(str(class_decoding(label[0].reshape(-1, nb_classes), nb_classes).reshape(-1)[0]))  # set title
        plt.imshow(img[0].astype(np.uint8), aspect='1.6', extent=[0, 1, 0, 1])
        ax=plt.gca()
        ax.get_xaxis().set_visible(False) 
        # yellow lines
        for y in np.arange(0.2, 0.8, 0.2):
                plt.axhline(y=y,color='yellow')
    plt.show()

def plot_acc_loss(history, modelname="modelname"):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(modelname)
    fig.set_figwidth(15)

    if "loss" in history.history:
        ax1.plot(history.history['loss'])
    if "accuracy" in history.history:
        ax2.plot(history.history['accuracy'])
    if "val_loss" in history.history:
        ax1.plot(history.history['val_loss'])
    if "val_accuracy" in history.history:
        ax2.plot(history.history['val_accuracy'])
    if "student_loss" in history.history:
        ax1.plot(history.history['student_loss'])
    if "categorical_accuracy" in history.history:
        ax2.plot(history.history['categorical_accuracy'])
    if "val_categorical_accuracy" in history.history:
        ax2.plot(history.history['val_categorical_accuracy'])
    if "student_accuracy" in history.history:
        ax2.plot(history.history['student_accuracy'])
    if "val_student_accuracy" in history.history:
        ax2.plot(history.history['val_student_accuracy'])
    if "distillation_loss" in history.history:
        ax1.plot(history.history['distillation_loss'])

    ax1.set_title('model loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax1.legend(['train','eval'], loc='upper left')
    axes = plt.gca()
    axes.set_ylim([0.92,1])
    plt.show()


def plot_divergence(divergationset, title1, nb_classes):
    fig = plt.figure(figsize=(40, 10))
    fig.suptitle(title1)
    plt.bar(np.arange (0, nb_classes/10, 0.1), divergationset, width=0.09, align='center')
    plt.ylabel('count')
    plt.xlabel('digit class')
    plt.xticks(np.arange(0, nb_classes/10, 0.1))
    return fig


def confusion_matrix(predicted, y_test, nb_classes):
    ytrue = pd.Series(y_test.reshape(-1), name = 'actual')
    ypred = pd.Series(predicted.reshape(-1), name = 'pred')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', None)
    return pd.crosstab(ytrue, ypred)


def predict_meter_digits(model, x_data, y_data, f_data, max_delta = 0.11):
    import numpy as np
    from tensorflow import keras

    predictions = class_decoding(model.predict(x_data.astype(np.float32)), 100).reshape(-1)

    # 9.9 <> 0 = 0.1 and 1.1 <> 1.2 = 0.1
    differences = np.minimum(np.abs(predictions-y_data), np.abs(predictions-(10-y_data)))

    # used for filtering
    false_differences = differences>max_delta

    # only differences bigger than delta. so small differences can be ignored in early stages
    false_predicted = differences[false_differences]
    false_images = x_data[false_differences]
    false_labels = [ "Expected: " + str(y) + "\n Predicted: " + str(p) + "\n" + str(f)[-26:-4] for y, p, f in zip(y_data[false_differences], predictions[false_differences], f_data[false_differences])]

    print(f"Tested images: {len(y_data)}. {len(false_predicted)} false predicted. Accuracy is: {1-len(false_predicted)/len(y_data)}")

    # plot the differences (max difference can only be 5.0)
    plot_divergence(np.bincount(np.array(false_predicted*10).astype(int), minlength=51), "Divergation of false predicted", 51)

    # plot the false predicted images
    plot_dataset(np.array(false_images), false_labels, columns=7, rows=7, figsize=(18,18))


def evaluate_ziffer_tflite(model_path, x_data, y_data, f_data, title, max_delta = 0.11):
    false_images = []
    false_labels = []
    false_predicted = []

    # we use the tflite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]


    for x, y, f in zip(x_data, y_data, f_data):
        
        interpreter.set_tensor(input_index, np.expand_dims(x.astype(np.float32), axis=0))
        # Run inference.
        interpreter.invoke()
        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.get_tensor(output_index)
        prediction = class_decoding(output)[0][0]
        difference = min(abs(prediction-y), abs(prediction-(10-y)))
        #print(prediction, y, difference)
        if difference>max_delta:
            false_images.append(x)
            false_labels.append( "Expected: " + str(y) + "\n Predicted: " + str(prediction) + "\n" + str(f)[-26:-4])
            false_predicted.append(difference)
               
    
    print(f"Tested images: {len(y_data)}. {len(false_labels)} false predicted. Accuracy is: {1-len(false_labels)/len(y_data)}")
    # plot the differences (max difference can only be 5.0)
    plot_divergence(np.bincount(np.array(np.array(false_predicted)*10).astype(int), minlength=51), "Divergation of false predicted", 51)

    # plot the false predicted images
    plot_dataset(np.array(false_images), false_labels, columns=7, rows=7, figsize=(18,18))

