# My first neural network (in progress)

## Abstract
My first neural network. I will try to represent, and train, from scratch, the multilayer perceptron presented in 3b1b explanation series on NN.

## How to run the tests
From the root directory, run `pytest` in the terminal. To run only one test script, run `pytest tests/<test_script_name>` from root.

## Overview of the folders
- `docs` brings together math notes and timing results of some functions.
- `src` contains the code.
- `tests`
- `playground.ipynb` is the only document one should interact with once the project is finished. In it, one can import the objects and functions necessary to instantiate and train a classifier.

## The dataset and how to download it.
The dataset used is a subset of the classical MNIST database of handwritten digits (28*28). It is retrieved from : https://www.kaggle.com/datasets/hojjatk/mnist-dataset. It has a training set of precisely 60,000 examples, and a test set of precisely 10,000 examples.
There's a dedicated function to download it : `download_dataset_from_internet` in `src/mnist_reader.py`.  

## AI usage
Through the completion of this project, AI is extensively used, and only used, to answer syntax questions or understanding and dealing with error messages. In particular, it is not used to write extensive portions of the code, as it would defeat the purpose of the project, i.e. learn through struggle.