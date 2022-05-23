from urllib.error import HTTPError
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



target_path = "/data"                   # root data path
target_raw_path =  "/data/raw_images"   # here all raw images will be stored
target_predicted_path = "/data/predicted" # here all predictions are copied 

# list of servernames of the edgeAI devices
meters = (  "gasmeter1", "watermeter1", "watermeter2","emeter1")

def yesterday(daysbefore=1):
    ''' return the date of yesterday as string in format yyyymmdd'''
    yesterday = date.today() - timedelta(days=daysbefore)
    return yesterday.strftime("%Y%m%d")


def readimages(servername, output_dir):
    '''get all images taken yesterday and store it in target path'''
    serverurl = "http://" + servername
    count = 0
    print(f"Loading data from {servername} ...")
    for datesbefore in range(1,5):
        picturedate = yesterday(daysbefore=datesbefore)
        # only if not exists already
        if not os.path.exists(path = output_dir + "/" + servername + "/" + picturedate):
            print("Loding ... ",  servername + "/" + picturedate)
            for i in range(24):
                hour = f'{i:02d}'
                try:
                    fp = urllib.request.urlopen(serverurl + "/fileserver/log/digit/" + picturedate + "/" + hour + "/")
                except HTTPError as h:
                    print("HTTPError at : " + serverurl + "/fileserver/log/digit/" + picturedate + "/" + hour + "/")
                    continue
                mybytes = fp.read()

                mystr = mybytes.decode("utf8")
                fp.close()

                urls = re.findall(r'href=[\'"]?([^\'" >]+)', mystr)
                path = output_dir + "/" + servername + "/" + picturedate + "/" + hour
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
        y_file = np.vstack((y_file, [aktfile]))
        x_data = np.vstack((x_data, test_image))
        y_data = np.vstack((y_data, [category]))
    print("Ziffer data count: ", len(y_data))   
    return x_data, y_data, y_file

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


from urllib.error import HTTPError
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

def remove_similar_images(image_filenames, hashfunc = imagehash.average_hash):
    '''removes similar images. 
    
    '''
    images = []
    count = 0
    cutoff = 3  # maximum bits that could be different between the hashes. 
    print(len(image_filenames))
  
    for img in sorted(image_filenames):
        try:
            hash = hashfunc(Image.open(img).convert('L'))
        except Exception as e:
            print('Problem:', e, 'with', img)
            continue
        images.append([hash, img])
    
    duplicates = {'1'}
    for hash in images:
        if (hash[1] not in duplicates):
            #print(hash[1])
            similarimgs = [i for i in images if abs(i[0]-hash[0]) < cutoff and i[1]!=hash[1]]
            #print(set([row[1] for row in similarimgs]))
            duplicates |= set([row[1] for row in similarimgs])
            imgstoshow = []
            labels = []
            for imgf in similarimgs:
                test_image = Image.open(imgf[1])
                test_image = np.array(test_image, dtype="float32")
                test_image = test_image/255.
                imgstoshow.append(test_image)
                labels.append(os.path.basename(imgf[1])[:-4])
    print("Duplicates: ", len(duplicates))
    for image in set(images)-duplicates:
        shutil.copy(image, os.path.join('test_data', os.path.basename(image)))

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

    model = keras.models.load_model("/models/eff100-gray.h5")

    for x, y, filename in zip(xz_data, yz_data, fz_data):
        base = os.path.basename(filename[0])
        #print(f"predict: ", filename)
        classes = model.predict(np.expand_dims(x.reshape(input_shape), axis=0))
        out_target = class_decoding(classes).reshape(-1)[0]
        target_filename = os.path.join(output_dir,str(out_target)+"_"+base)
        # do not overwrite files
        if not os.path.exists(target_filename):
            shutil.copy(filename[0], target_filename)

def mark_duplicates(input_dir):
    ''' mark all duplicated (different predicted)  images with "X" at prefix'''
    filenames = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if (file.endswith(".jpg")):
                filenames.append((root + "/" + file, file))
    for filename in filenames:
        file = filename[1]

        # do not mark multiple times
        if (file[0] == 'X'):
            print("con")
            continue

        # if predicted, remove prediction for search
        if (file[1] == '.'):
            file = file[4:]
        
        print("Filename=", file)
        # search for substrings
        for s in filenames:
            print(s, file)
            if (s[1].endswith(file) and not( file == s[1])):
                os.rename(filename[0], os.path.dirname(filename[0]) + "/X" +os.path.basename(filename[0]))
       

# ensure the target path exists
print("retrieve and prepare")
os.makedirs(target_raw_path, exist_ok=True)
os.makedirs(target_predicted_path, exist_ok=True)

# read all images from meters
for meter in meters:
    readimages(meter, target_raw_path)
    print("remove now all similar images")
    # remove all same or similar images and remove the empty folders
    remove_similar_images(ziffer_data_files(os.path.join(target_raw_path, meter)))



# predict now the images
#print("predict now all images and move to predicted directory")
#predict_images(target_raw_path, target_predicted_path)

# mark now duplicated images (manual fix or new model predicts newly)
#print("mark all multiple and different predicted images with X")
#mark_duplicates(target_predicted_path)

#remove emtpy folders
#print("cleanup all directories in raw_path")
#remove_empty_folders(target_raw_path)




