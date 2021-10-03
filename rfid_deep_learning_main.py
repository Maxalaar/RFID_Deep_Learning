from general_include import *

from recurrent_dataset import * 
from recurrent_neural_network_classifier import *
from classic_dataset import *
from dense_neural_network_classifier import *

environment_info()

# create_tracking_dataset(2, 3, "debug_dataset", "debug_reccurent_simple_dataset", 2, num_repetitions_target=1, num_repetitions_zone=1, verbose=True)

create_tracking_dataset(10, 19, "kitchen_20cm", "debug_reccurrent_kitchen_20cm", 2, num_repetitions_target=1, num_repetitions_zone=1, verbose=True)

# data_X, data_Y = load_tracking_dataset("debug_reccurent_dataset")

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