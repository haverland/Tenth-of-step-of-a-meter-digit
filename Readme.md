# Project goal

The project tries to recognize all digits of gas, water and electric meters in percental pieces if the digit is moving to the next digit.

The idea comes from <https://github.com/jomjol/AI-on-the-edge-device> project.

## Project structure

Folders:
*datasets - TMNIS-Dataset and the TMNIS-percental dataset
*failures - after run all failures of meter digits will be moved here for later manual post processing
*images - all labeled meter digits.
*prepare_data - all pre processing notebooks for creating the percental TMNIS-Digits, sorting out not good fonts, retrieving images from my own AI-on-the-edge-devices
*raw_images - stored retrieved images from my own AI-on-the-edge-devices
*test_data - all images for the test_manual - notebook.

## Datasets

### Percental TMNIST

The Type face MNIST dataset comes from kaggle <https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist> and is processed for percental digits <https://www.kaggle.com/datasets/frankhaverland/percentile-tmnist-for-electric-meters>.

The complete dataset has 300.000 images. Some fonts are not good for training the meter digits and will removed by blacklist entries.

### Images of gas, water and power meters

The folder contains images of <https://github.com/jomjol/neural-network-autotrain-digital-counter/tree/main/ziffer_raw> and collected images from others. See <https://github.com/haverland/collectmeterdigits>.
