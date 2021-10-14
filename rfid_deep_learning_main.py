from general_include import *

from recurrent_dataset import * 
from recurrent_neural_network_classifier import *
from classic_dataset import *
from dense_neural_network_classifier import *

environment_info()

# create_tracking_dataset(2, 3, "debug_dataset", "debug_reccurent_simple_dataset", 2, num_element_path=5, num_repetitions_zone=1, verbose=True)
# create_tracking_dataset(10, 19, "kitchen_20cm", "debug_reccurrent_kitchen_20cm", 10, num_element_path=5, num_repetitions_zone=1, verbose=True)
# create_tracking_dataset(10, 19, "kitchen_20cm", "test__reccurrent_kitchen_20cm", 20000, num_element_path=30, num_repetitions_zone=1, verbose=False)

# create_tracking_dataset(10, 19, "kitchen_20cm", "test_v2_reccurrent_kitchen_20cm", 5000, num_element_path=300, num_repetitions_zone=(20, 40), verbose=False)
# create_tracking_dataset(10, 19, "kitchen_20cm", "test_v3_reccurrent_kitchen_20cm", 100, num_element_path=300, num_repetitions_zone=(20, 40), verbose=False)

# RNN
if False :
    model = None
    # model = load_model_neural_network("RNN_3")

    # data_X, data_Y = load_tracking_dataset("test_v3_reccurrent_kitchen_20cm")
    # model = recurrent_neural_network_classifier(data_X, data_Y, model=model)
    # save_model_neural_network(model, "RNN_4_Noise")

    # evaluation_dataset = load_tracking_dataset_txt("kit_a")[1]
    # evaluation_dataset = [[[0.0, -65.0, -58.0, -66.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -65.0, -65.0, -63.0, -64.0]]]
    # evaluate_rnn_model(model, evaluation_dataset)
    # model.evaluate(data_X, data_Y)

# NN
if False :
    goodtechs_np_X, goodtechs_np_Y, label_encoder_goodtechs = load_classic_dataset("kitchen_20cm")
    model = dense_neural_network_classifier(goodtechs_np_X, goodtechs_np_Y)
    save_model_neural_network(model, "DNN_1")

#
if True :
    model = None
    # data_X, data_Y = load_tracking_dataset_txt_for_fit(["kit_a", "kit_b", "kit_c", "kit_d"])
    data_X, data_Y = load_tracking_dataset_txt_for_fit(["kit_a"])
    print(data_X.shape)
    print(data_Y.shape)
    model = recurrent_neural_network_classifier(data_X, data_Y)