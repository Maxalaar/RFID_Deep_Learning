import os
import sys
import tensorflow as tf

import pickle
import pandas as pd
import numpy as np
import random
from scipy.io import arff

from sklearn import preprocessing

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Dropout
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def environment_info():    
    print()
    print("--- Environment Information ---")
    print()

    # Allows to not  display the debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    print("Path of execution : " + os.getcwd())
    print()

    print("Python version : " + str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "." + str(sys.version_info[2]))
    print()

    print("Tensorflow version : " + str(tf.__version__))
    print()

    print("Number of GPU available for Tensorflow :", len(tf.config.list_physical_devices('GPU')))
    print()

def save_model_neural_network(model, model_name):
    model.save('./saved_model_nn/' + model_name)

def load_model_neural_network(model_name, verbose=True):
    print()
    print("--- Load Neural Network ---")
    print()

    model = tf.keras.models.load_model('./saved_model_nn/' + model_name, custom_objects={'f1_m':f1_m})

    if verbose == True:
        model.summary()

    return model

def load_checkpoint_neural_network(model, checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)