from general_include import *

from recurrent_dataset import * 
from recurrent_neural_network_classifier import *
from classic_dataset import *
from dense_neural_network_classifier import *
from classic_algorithms_big_data import *
from conv_neural_network_classifier import *

environment_info()

# create_tracking_dataset(2, 3, "debug_dataset", "debug_reccurent_simple_dataset", 2, num_element_path=5, num_repetitions_zone=1, verbose=True)
# create_tracking_dataset(10, 19, "kitchen_20cm", "debug_reccurrent_kitchen_20cm", 10, num_element_path=5, num_repetitions_zone=1, verbose=True)
# create_tracking_dataset(10, 19, "kitchen_20cm", "test__reccurrent_kitchen_20cm", 20000, num_element_path=30, num_repetitions_zone=1, verbose=False)

# create_tracking_dataset(10, 19, "kitchen_20cm", "test_v2_reccurrent_kitchen_20cm", 5000, num_element_path=300, num_repetitions_zone=(20, 40), verbose=False)
# create_tracking_dataset(10, 19, "kitchen_20cm", "test_v3_reccurrent_kitchen_20cm", 100, num_element_path=300, num_repetitions_zone=(20, 40), verbose=False)

# RNN
if True :
    X_data, Y_data = create_tracking_dataset(10, 19, "kitchen_20cm", "debug_reccurrent_kitchen_20cm", 666, num_element_path=300, num_repetitions_zone=(20, 40), verbose=False)
    model = recurrent_neural_network_classifier_synt_tracking_dataset(X_data, Y_data)

if False :
    # RNN_2 in dataset V2 = 95%
    model = None
    model = load_model_neural_network("RNN_2")

    data_X, data_Y = load_tracking_dataset("v1_100_300_reccurrent_kitchen_20cm")
    # data_X, data_Y = load_tracking_dataset("test_v2_reccurrent_kitchen_20cm")
    # data_X, data_Y = load_tracking_dataset_txt_for_fit(["kit_a", "kit_b", "kit_c", "kit_d"])

    model = recurrent_neural_network_classifier_for_text_dataset(data_X, data_Y)
    # model = recurrent_neural_network_classifier_synt_tracking_dataset(data_X, data_Y)
    # model = recurrent_neural_network_classifier_synt_tracking_dataset(data_X, data_Y, model=model)
    # save_model_neural_network(model, "RNN_4_Noise")

    # evaluation_dataset = load_tracking_dataset_txt("kit_a")[1]
    # evaluation_dataset = [[[0.0, -65.0, -58.0, -66.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -65.0, -65.0, -63.0, -64.0]]]
    # evaluate_rnn_model(model, evaluation_dataset)
    
    model.evaluate(data_X, data_Y)

# NN
if False :
    goodtechs_np_X, goodtechs_np_Y, label_encoder_goodtechs = load_classic_dataset("kitchen_20cm")
    model = dense_neural_network_classifier(goodtechs_np_X, goodtechs_np_Y)
    save_model_neural_network(model, "DNN_1")

# 
if False :
    model = None
    data_X, data_Y = load_tracking_dataset_txt_for_fit(["kit_a", "kit_b", "kit_c", "kit_d"])
    # data_X, data_Y = load_tracking_dataset_txt_for_fit(["kit_a"])
    model = recurrent_neural_network_classifier_for_text_dataset(data_X, data_Y)
    # model = recurrent_neural_network_classifier_synt_tracking_dataset(data_X, data_Y)

# Tree for txt
if False :
    # create_tracking_dataset(10, 19, "kitchen_20cm", "v1_100_300_reccurrent_kitchen_20cm", 3000, num_element_path=368, num_repetitions_zone=(20, 40), verbose=False)

    data_X, data_Y = load_tracking_dataset("v1_100_300_reccurrent_kitchen_20cm")
    # data_X, data_Y = convert_recurrent_dataset_to_classic_dataset(data_X, data_Y)

    model = recurrent_neural_network_classifier_synt_tracking_dataset(data_X, data_Y)
    # model = dense_neural_network_classifier(data_X, data_Y)
    # model = k_neighbors_classifier(data_X, data_Y, 1)
    # model = decision_trees_classifier(data_X, data_Y)
    # model = random_forest_classifier(data_X, data_Y)

    # data_X, data_Y = load_tracking_dataset_txt_for_fit(["kit_a", "kit_b", "kit_c", "kit_d"])
    # data_X, data_Y = convert_recurrent_dataset_to_classic_dataset(data_X, data_Y)

    model.evaluate(data_X, data_Y)
    # print("Score : " + str(model.score(data_X, data_Y)))

    # data_X, data_Y = load_tracking_dataset("v1_100_300_reccurrent_kitchen_20cm")

    # k_neighbors_classifier(data_X, data_Y, 1)
    # decision_trees_classifier(data_X, data_Y)
    # random_forest_classifier(data_X, data_Y)

if False :
    data_X, data_Y = load_tracking_dataset_txt_for_fit(["kit_a", "kit_b", "kit_c", "kit_d"])
    # data_X, data_Y = convert_recurrent_dataset_to_classic_dataset(data_X, data_Y)
    # model = dense_neural_network_classifier(data_X, data_Y)
    model = recurrent_neural_network_classifier_for_text_dataset(data_X, data_Y)

if False :
    # data_X, data_Y, encodeur = load_classic_dataset("kitchen_20cm")
    data_X, data_Y, encodeur = load_classic_dataset("goodtechs_data")

    # model = dense_neural_network_classifier(data_X, data_Y)
    # model = load_model_neural_network("dense_RNN_1")
    # neural_conv_network_classifier(data_X, data_Y)

    k_neighbors_classifier(data_X, data_Y, 5)