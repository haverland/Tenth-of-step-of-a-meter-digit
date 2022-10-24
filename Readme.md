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

It compares two times. First with delta +/- 0.1 as ok and the second without any delta. It is because the models not reaches >99% accuracy without delta, but in most times with delta=0.1
