# My first neural network (in progress)

## Abstract
My first neural network. I represent, and train, from scratch (=numpy only), the multilayer perceptron presented in 3b1b explanation series on NN. 

## How to build the project
This project uses a few libraries. To add them to the environment, from the root directory, run `pip install .`. They are listed in the `pyproject.toml` file. You might also install them manually.

## How to run the tests
From the root directory, run `pytest` in the terminal. To run only one test script, run `pytest tests/<test_script_name>` from root.

## Overview of the folders
- `docs` brings together math notes and plots higlighting interesting properties.
- `src` contains the code.
- `tests`
- `playground.ipynb` is the only document one should interact with once the project is finished. In it, one can import the objects and functions necessary to instantiate and train a classifier.
- `saved_ojects` is useful if you want to save objects for use elsewhere or in another jupyter notebook session. Custom save and load functions are presented in `playground.ipynb`.

## The dataset and how to download it.
The dataset used is a subset of the classical MNIST database of handwritten digits (28*28). It is retrieved from : https://www.kaggle.com/datasets/hojjatk/mnist-dataset. It has a training set of precisely 60,000 examples, and a test set of precisely 10,000 examples.
There's a dedicated function to download it : `download_dataset_from_internet` in `src/mnist_reader.py`.  

## AI usage
Through the completion of this project, AI is extensively used, and only used, to answer syntax questions or understanding and dealing with error messages. In particular, it is not used to write more than one-liners, as it would defeat the purpose of the project, i.e. learn through struggle.