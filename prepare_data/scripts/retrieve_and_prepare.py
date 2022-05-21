import urllib.request
import re
import requests
import os
from PIL import Image
from datetime import date, timedelta
import shutil
import numpy as np
import imagehash
import tensorflow.keras as keras



target_path =  "/raw_images"
meters = (  "gasmeter1", "watermeter1", "watermeter2","emeter1")

def yesterday():
    ''' return the date of yesterday as string in format yyyymmdd'''
    yesterday = date.today() - timedelta(days=1)
    return yesterday.strftime("%Y%m%d")


def readimages(servername):
    '''get all images taken yesterday and store it in target path'''
    serverurl = "http://" + servername
    count = 0
    print(f"Loading data from {servername} ...")
    for i in range(24):
        hour = f'{i:02d}'
        #(serverurl + "/fileserver/log/digit/" + yesterday() + "/" + hour+ "/")
        fp = urllib.request.urlopen(serverurl + "/fileserver/log/digit/" + yesterday() + "/" + hour + "/")
        mybytes = fp.read()

        mystr = mybytes.decode("utf8")
        fp.close()

        urls = re.findall(r'href=[\'"]?([^\'" >]+)', mystr)
        path = target_path + "/" + servername + "/" + yesterday() + "/" + hour
        os.makedirs(path, exist_ok=True) 
        for url in urls:
            filename = os.path.basename(url)
            if (not os.path.exists(path + "/" + filename)):
                #print(serverurl+url)
                img = Image.open(requests.get(serverurl+url, stream=True).raw)
                img.save(path + "/" + filename)
                count = count +1
    print(f"{count} images are loaded from meter: {servername}")



def ziffer_data_files(input_dir):
    '''return a list of all images in given input dir in all subdirectories'''
    imgfiles = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if (file.endswith(".jpg")):
                imgfiles.append(root + "/" + file)
    return  imgfiles


def ziffer_data(input_dir, use_grayscale=True):
    '''return a tuple of (images, labels, filenames) in the given input dir'''
    y_file = []
    y_data = []
    x_data = []
    files = ziffer_data_files(input_dir)
    y_data = np.array(y_data).reshape(-1,1)
    y_file = np.array(y_file).reshape(-1,1)
    if (use_grayscale):
        x_data = np.array(x_data).reshape(-1,32,20,1)
    else: 
        x_data = np.array(x_data).reshape(-1,32,20,3)

    for aktfile in files:
        base = os.path.basename(aktfile)
        if (base[1]=="."):
            target = base[0:3]
        else:
            target = base[0:1]
        if target == "N":
            category = 10                # NaN does not work --> convert to 10

        else:
            category = float(target)
        test_image = Image.open(aktfile).resize((20, 32))
        if (use_grayscale):
            test_image = test_image.convert('L')
        test_image = np.array(test_image, dtype="float32")
        test_image = test_image/255.
            
        #print(test_image.shape)
        if (use_grayscale):
            test_image = test_image.reshape(1,32,20,1)
        else:
            test_image = test_image.reshape(1,32,20,3)

        # ignore the category 10
        #if ( category<10):
        y_file = np.vstack((y_file, [base]))
        x_data = np.vstack((x_data, test_image))
        y_data = np.vstack((y_data, [category]))
    print("Ziffer data count: ", len(y_data))   
    return x_data, y_data, y_file

def move_to_pred_dir(prediction, filename, input_dirs=['test_data'], out_dir='predicted'):
    for input_dir in input_dirs:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.startswith(filename):
                    shutil.move(os.path.join(root, file), os.path.join(out_dir,str(prediction)+"_"+file))


def class_encoding(y_train, nb_classes):
    ''' like to_categorical in sorted order'''
    ret = np.zeros((len(y_train), nb_classes))
    for i, y in enumerate(y_train):
        ret[i, int((y*10))] = 1
    return ret

def class_decoding(y_train, nb_classes=100):
    ''' like argmax but the return value is between 0.0 and 9.9'''
    ret = np.zeros((len(y_train), 1))
    for i, y in enumerate(y_train):
        ret[i] = (np.argmax(y))/10
    return ret


def remove_similar_images(image_filenames, hashfunc = imagehash.average_hash, hash_size=3):
    '''removes similar images. 
    hash_size 2..n smaller values for more detected images
    '''
    images = {}
    count = 0
    
    for img in sorted(image_filenames):
        try:
            hash = hashfunc(Image.open(img), hash_size=hash_size)
        except Exception as e:
            print('Problem:', e, 'with', img)
            continue
        if hash in images:
            #print(img, '  already exists as', ' '.join(images[hash]))
            for dup in images[hash]:
                try:
                    os.remove(img)
                    count=count+1
                except Exception as e1:
                    continue
        images[hash] = images.get(hash, []) + [img]
    print(f"{count} removed.")


def remove_empty_folders(path_abs):
    '''all empty folders in path_abs will be deleted. not the path_abs'''
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0 and not path == path_abs:
            print("remove: ", path)
            os.rmdir(path)

def predict_images(input_dir, output_dir):
    '''predict all images input_dir and move the images to output_dir.
       The filename begins with prediction.
    '''
    
    xz_data, yz_data, fz_data = ziffer_data(input_dir,  use_grayscale=True)
    input_shape=xz_data[0].shape


    model = keras.models.load_model("/model/eff100-gray.h5")

    os.makedirs(output_dir, exist_ok=True)

    for x, y, filename in zip(xz_data, yz_data, fz_data):

        classes = model.predict(np.expand_dims(x.reshape(input_shape), axis=0))
        out_target = class_decoding(classes).reshape(-1)[0]
        move_to_pred_dir(out_target, filename[0], out_dir=output_dir )



# ensure the target path exists
os.makedirs(target_path, exist_ok=True)

# read all images from meters
for meter in meters:
    readimages(meter)

# remove all same or similar images and remove the empty folders
remove_similar_images(ziffer_data_files(target_path))

# predict now the images
predict_images(target_path, target_path)

#remove emtpy folders
remove_empty_folders(target_path)




