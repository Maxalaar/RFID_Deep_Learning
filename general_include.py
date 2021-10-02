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

def environment_info():
    print()
    print("--- Environment Information ---")
    print()

    print("Path of execution : " + os.getcwd())
    print()

    print("Python version : " + str(sys.version_info))
    print()

    print("Number of GPU available for Tensorflow :", len(tf.config.list_physical_devices('GPU')))
    print()