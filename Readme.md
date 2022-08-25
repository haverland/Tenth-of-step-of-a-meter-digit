# Project goal

The project tries to recognize all digits of gas, water and electric meters in percental pieces if the digit is moving to the next digit.

The idea comes from <https://github.com/jomjol/AI-on-the-edge-device> project.

## Project structure

Folders:

* datasets - TMNIS-Dataset and the TMNIS-percental dataset
* failures - after run all failures of meter digits will be moved here for later manual post processing
* images - all labeled meter digits.
* prepare_data - all pre processing notebooks for creating the percental TMNIS-Digits, sorting out not good fonts, retrieving images from my own AI-on-the-edge-devices
* raw_images - stored retrieved images from my own AI-on-the-edge-devices
* test_data - all images for the test_manual - notebook.

## Datasets

### Percental TMNIST

The Type face MNIST dataset comes from kaggle <https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist> and is processed for percental digits <https://www.kaggle.com/datasets/frankhaverland/percentile-tmnist-for-electric-meters>.

The complete dataset has 300.000 images. Some fonts are not good for training the meter digits and will removed by blacklist entries.

### Images of gas, water and power meters

The folder contains images of <https://github.com/jomjol/neural-network-autotrain-digital-counter/tree/main/ziffer_raw> and collected images from others. See <https://github.com/haverland/collectmeterdigits>.

## Naming and Versioning

The naming of the notebooks is `dig-class<output>_<size>.ipynb`.

* *output* can be
  * `dig-class10` for categorical (`dig` as model name)
* size 
  s0: >30 M FLOPS
  s1: 20-30 M FLOPS
  s2: 10-20 M FLOPS
  s3: < 10 M FLOPS
  
### Transfer learning

On Transfer learning a model will be trained with other datasets and learn the convolutional layers.
Mostly pretrained models are used. We learn the model with tenth of step of TMNIST dataset (folder dataset) and the meter digit images.

In a second step only the last layer or all fully connected layers are retrained. The convolutional layers will not be trained.

Examples are `dig-class100_s1_transfer`

### Learning on meter digits

The notebooks learning only on meter digit images as long as the model can be trained. The quality depends on the mount of images. (Currently 12.000)

After run a csv file will created with list of false predicted image file names. The file can be used with

   python3 -m collectmeterdigits --labelfile=output/eff100md_false_predicted.csv

to fix labels or check the labeling.

### Quantization

For using on edgeAI devices the model will be quantized. It can be used with Tensorflow light.

The notebook `make_tflite.ipynb` creates the tflite-model and evaluate the results. This are the results of the target edgeAI device.

### Comparing the different models

To get a better overview of the different models and their results, the notebook compare_all_tflite.ipynb can be used.

It can handle classification models with output of 100 classes and the hyprid models with 10 classes too.

Older models with 11 classes are not comparable.

It compares two times. First with delta +/- 0.1 as ok and the second without any delta. It is because the models not reaches >99% accuracy without delta, 
but in most times with delta=0.1

## Versions

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
