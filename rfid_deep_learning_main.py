from general_include import *

from recurrent_dataset import * 
from recurrent_neural_network_classifier import *
from classic_dataset import *
from dense_neural_network_classifier import *

environment_info()

# create_tracking_dataset("kitchen_20cm", "debug_kitchen_20cm", 10, 1, 1, 2, 3, verbose=False)

# data_X, data_Y = load_tracking_dataset("debug_kitchen_20cm")
# data_X, data_Y = load_tracking_dataset("test_kitchen_20cm")
# recurrent_neural_network_classifier(data_X, data_Y)

# data_X, data_Y, label_encoder_goodtechs = load_classic_dataset("kitchen_20cm")
# dense_neural_network_classifier(data_X, data_Y, verbose=[])

# print(data_X)
# print(data_Y)
# print(data_X[0][25])
# print(data_Y[0][25])
# # print(data_X[100][58])
# # print(data_X[200][30])
# # print(type(data_X))