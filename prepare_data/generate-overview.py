import os
import math
from PIL import Image, ImageOps
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)


def concat_images(image_paths, size, shape=None):
    # Open images and resize them
    width, height = size
   # width = width * 2
    images = map(Image.open, image_paths)
    
    images = [ImageOps.expand(image, border=2,fill='white')
              for image in images]
    
    images = [ImageOps.fit(image, size, Image.ANTIALIAS) 
              for image in images]
    
    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (int(width * shape[1]), int(height * shape[0]))
    image = Image.new('RGB', image_size, color='white')
    
    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):            
            offset = width * col, height * row
            idx = row * shape[1] + col
            try:
                image.paste(images[idx], offset)
            except:
                pass
    
    return image


def generate(path, prefix, cols):
    logging.info("Generating summary image of all '%s*.jpg' images in %s..." % (prefix, path))
    # Get list of image paths
    
    image_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if (file.endswith(".jpg") and file.startswith(prefix)):
                #print(root + "/" + file)
                image_paths.append(root + "/" + file)
    

    #image_paths = image_paths[:11]

    logging.debug("Found %d images." % len(image_paths))


    rows = math.ceil(float(len(image_paths)) / cols)

    logging.debug("Generating grid of %d x %d images" % (cols, rows))


    # Create and save image grid
    image = concat_images(image_paths, (int(800/cols), int(800/cols)), (rows, cols))
    os.makedirs("./output/html/", exist_ok=True)
    image.save("./output/html/dig-" + prefix + ".jpg", 'JPEG')




values = ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"]
path = './images/'

for value in values:
    generate(path, value, 10)