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

## Notebooks and strategies

The naming of the notebooks is `pmd-<output>_<model>_<strategy>.ipynb`.

* *output* can be
  * `cat`for categorical
  * `reg`for regression
* *model* are a short name for the model type
  * `eff`for Effnet <https://github.com/arthurdouillard/keras-effnet>
  * `cnn`classical cnn model. Can be different in size.
* *strategy*
  * `transfer` for transfer learning
  * `teacher`for knowledge distillation the teacher model
  * `distillation` for knowledge distillation of the student model.

### Transfer learning

On Transfer learning a model will be trained with other datasets and learn the convolutional layers.
Mostly pretrained models are used. We learn the model with tenth of step of TMNIST dataset (folder dataset) and the meter digit images.

In a second step only the last layer or all fully connected layers are retrained. The convolutional layers will not be trained.

Examples are `pmd-cat_eff_transfer` and `pmd-cat_CNN3_transfer`.

### Knowledge distillation

A bigger teacher model will be trained with a real good accuracy. After it a smaller student model will be trained with help of the teacher model. For more details <https://keras.io/examples/vision/knowledge_distillation/>

Examples are `pmd-cat_CNN_teacher`and `pmd-cat_eff_distillation`.

Because of the good results in transfer learning a knowledge distillation does not have a great effect.

### Pruning and Quantization

For using on edgeAI devices the model will be pruned and quantized. It can be used with Tensorflow light.

The notebook `make_tflite.ipynb` creates the tflite-model and evaluate the results. This are the results of the target edgeAI device.
