import urllib.request
import re
import requests
import os
import requests
from PIL import Image
from datetime import datetime

target_path =  "/raw_images"
meters = (  "gasmeter1", "watermeter1", "watermeter2","emeter1")

def yesterday():
    from datetime import date, timedelta
    yesterday = date.today() - timedelta(days=1)
    return yesterday.strftime("%Y%m%d")


def readimages(servername):
    serverurl = "http://" + servername
    
    for i in range(24):
        hour = f'{i:02d}'
        print(serverurl + "/fileserver/log/digit/" + yesterday() + "/" + hour+ "/")
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
                print(serverurl+url)
                img = Image.open(requests.get(serverurl+url, stream=True).raw)
                img.save(path + "/" + filename)

from PIL import Image
import six
import shutil
import os

import imagehash

def ziffer_data_files(input_dir='/raw_images'):
    imgfiles = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if (file.endswith(".jpg")):
                #print(root + "/" + file)
                imgfiles.append(root + "/" + file)
    return  imgfiles


def remove_similar_images(image_filenames, hashfunc = imagehash.average_hash):
    images = {}
    for img in sorted(image_filenames):
        try:
            hash = hashfunc(Image.open(img))
        except Exception as e:
            print('Problem:', e, 'with', img)
            continue
        if hash in images:
            print(img, '  already exists as', ' '.join(images[hash]))
            for dup in images[hash]:
                try:
                    os.remove(img)
                except Exception as e1:
                    continue
        images[hash] = images.get(hash, []) + [img]

def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.remove(path)

os.makedirs(target_path, exist_ok=True)
for meter in meters:
    readimages(meter)

remove_similar_images(ziffer_data_files(target_path))
remove_empty_folders(target_path)


