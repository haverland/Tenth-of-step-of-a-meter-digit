# Project goal

The project tries to recognize all digits of gas, water and electric meters in percental pieces if the digit is moving to the next digit.

The idea comes from <https://github.com/jomjol/AI-on-the-edge-device> project.

## Project structure

Folders:

* images - all labeled meter digits.
* prepare_data - all pre processing notebooks for creating the percental TMNIS-Digits, sorting out not good fonts, retrieving images from my own AI-on-the-edge-devices
* output - the resulting tflite-models
* b2n - python util classes for smaller notebooks

## Datasets

### Images of gas, water and power meters

The folder `images` contains images of <https://github.com/jomjol/neural-network-autotrain-digital-counter/tree/main/ziffer_raw> and collected images from others. See <https://github.com/haverland/collectmeterdigits>.

An overview of the data can be found [here](https://github.com/haverland/Tenth-of-step-of-a-meter-digit/index.html).

#### Add my own data

The images can be collected with [collectmeterdigits](https://github.com/haverland/collectmeterdigits). Read the instructions of the project.

In `images/collected` can new images be added. The structure is

```
images
└───collected
    └───powermeter
    │    └ Manufacturer
    │      └ <yourshortcut>
    │         └ images.jpg
    └───watermeter
    │    └ Manufacturer
    │      └ <yourshortcut>
    │         └ images.jpg
    └───gasmeter
         └ Manufacturer
           └ <yourshortcut>
              └ images.jpg
```

## Naming and Versioning

The naming of the notebooks is `dig-class<output>_<size>.ipynb`.

* *output* can be
  * `dig-class10` for categorical (`dig` as model name)
* size 
  s0: >30 M FLOPS
  s1: 20-30 M FLOPS
  s2: 10-20 M FLOPS
  s3: < 10 M FLOPS

So dig-class100_s1 is bigger than dig-class100-s2
  
## Learning on meter digits

The notebooks learning only on meter digit images as long as the model can be trained. The quality depends on the mount of images. (Currently 17.000)

Add your images like described above and run 

dig-class100_s1.ipynb or
dig-class100_s2.ipynb

dig-class100_s0 is to big for the esp32 device and only used for comparisations.

After run a csv file will created with list of false predicted image file names. The file can be used with

   python3 -m collectmeterdigits --labelfile=output/tmp/dig-class100-s2_false_predicted.csv

to fix labels or check the labeling.


### Comparing the different models

To get a better overview of the different models and their results, the notebook `compare_all_tflite.ipynb` can be used.

It can handle classification models with output of 100 classes and the hyprid models with 10 classes too.

Older models with 11 classes are not comparable.


It compares two times. First with delta +/- 0.1 as ok and the second without any delta. It is because the models not reaches >99% accuracy without delta, 
but in most times with delta=0.1

## Versions

### 1.8.0
* upgrade to tensorflow 2.17
* breaking changes in preprocessing

### 1.7.0

* fix tensorflow vulnerability
* new images 
* fix kaggle upload

### 1.6.1 (2023-02-26)
* new images (lcd)
* fixed quantization bug - use all images for dataset
* dig-class-161_s2

### 1.6 (2022-12-27)
* new images
* dig-class-160_s2

### 1.5 (2022-12-18)
* new images
* dig-class-0150_s2

### 1.3 (2022-08-25)
* added new images
* dig-class-0130_s2 - with 99.9% accuracy (+/- 0.1) and 89% accuracy.

### 1.2 (2022-07-24)

* dig-class100_0120_s2 - with 99.8% accuracy (+/- 0.1) and 89.5% accuracy.
* compare_all_tflite supports dhy models

### 1.1 (not releases)

* dig-class100_0110_s1 - with 99.7% accuracy (+/- 0.1) and 88.7% accuracy.

### 1.0 (2022-07-16)

* dig-class100_0100_s2 - with 99.6% accuracy (+/-0.1)
