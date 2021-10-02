from general_include import *

def create_tracking_dataset(name_dataset, name_save, num_element, min_num_repetitions_target=1, max_num_repetitions_target=1, min_num_repetitions_zone=1, max_num_repetitions_zone=1, verbose=False):
    print()
    print("--- Create Tracking Dataset ---")
    print()

    # Load the dataset
    dataset_data = arff.loadarff('/home/maxalaar/CodiumProjects/RFID_Deep_Learning/dataset/'+str(name_dataset)+'.arff')
    dataset_complete = []

    # Convert the dataset, arff to DataFram
    dataset_df = pd.DataFrame(dataset_data[0])

    # Convert Y to one hot
    Y_one_hot = pd.get_dummies(dataset_df["class"])
    list_target_name = list(Y_one_hot.columns)
    dataset_df = dataset_df.copy().drop("class",axis=1)
    dataset_df = pd.concat([dataset_df, Y_one_hot], axis=1)

    # Creat map of zone
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    num_colonne = 10 # 14 for all room
    num_row = 19

    map_zone = []
    for i in range(0, num_colonne):
        row_prov = []

        for j in range(0, num_row):
            row_prov.append(None)
        
        map_zone.append(row_prov)
    
    # For each zone is given its dataset
    for i in range(0, num_colonne):
        for j in range(0, num_row):
            map_zone[i][j] = dataset_df.loc[dataset_df[str(alphabet[i] + str(j+1)).encode()] == 1]

    for _ in range (0, num_element):
        # We choose a init zone and target zone at random for creat list of zone
        if min_num_repetitions_target != max_num_repetitions_target:
            num_repetitions_target = random.randrange(min_num_repetitions_target, max_num_repetitions_target+1, 1)
        else:
            num_repetitions_target = max_num_repetitions_target
        
        list_zone_x = []
        list_zone_y = []

        frame = []
        data = None
        
        x_zone_init = random.randrange(0, num_colonne, 1)
        y_zone_init = random.randrange(0, num_row, 1)

        x_zone = x_zone_init
        y_zone = y_zone_init

        for i in range(0, num_repetitions_target):

            x_zone_target = random.randrange(0, num_colonne, 1)
            y_zone_target = random.randrange(0, num_row, 1)

            # add zone in zone list
            list_zone_x.append(x_zone)
            list_zone_y.append(y_zone)

            while x_zone != x_zone_target or y_zone != y_zone_target:

                num_ramdom = random.randrange(0, 100, 1)

                # move in x
                if num_ramdom < 33 and x_zone != x_zone_target:

                    if x_zone < x_zone_target:
                        x_zone += 1
                    
                    elif x_zone > x_zone_target:
                        x_zone -= 1
                
                # move in y
                elif num_ramdom < 66 and y_zone != y_zone_target:
                
                    if y_zone < y_zone_target:
                        y_zone += 1
                    
                    elif y_zone > y_zone_target:
                        y_zone -= 1
                
                # move in x and y
                else:
                    
                    if x_zone < x_zone_target:
                        x_zone += 1
                    
                    elif x_zone > x_zone_target:
                        x_zone -= 1

                    if y_zone < y_zone_target:
                        y_zone += 1
                    
                    elif y_zone > y_zone_target:
                        y_zone -= 1
                
                # add zone in zone list
                list_zone_x.append(x_zone)
                list_zone_y.append(y_zone)
        
        if verbose:
            print("Zone x : " + str(list_zone_x))
            print("Zone y : " + str(list_zone_y))

        # for each zone give some measure
        for i in range(0, len(list_zone_x)):
            num_repetitions_zone = random.randrange(min_num_repetitions_zone, max_num_repetitions_zone+1, 1)
            
            for j in range(0, num_repetitions_zone):
                frame.append(map_zone[list_zone_x[i]][list_zone_y[i]].sample())

        # each 
        data = pd.concat(frame)
        dataset_complete.append(data)

    # merge dataset
    dataset_complete_X_pandas = []
    dataset_complete_Y_pandas = []

    for data in dataset_complete:
        dataset_complete_X_pandas.append(data.copy().drop(list_target_name, axis=1))
        dataset_complete_Y_pandas.append(data[list_target_name])

    # Convert Pandas dataset to Python list
    dataset_complete_X_list = []
    dataset_complete_Y_list = []

    for i in range(0, len(dataset_complete_Y_pandas)):
        dataset_complete_X_list.append(dataset_complete_X_pandas[i].values.tolist())
        dataset_complete_Y_list.append(dataset_complete_Y_pandas[i].values.tolist())

    # Convert Python list to Tensorflow Tensor
    X_tensor = tf.ragged.constant(dataset_complete_X_list).to_tensor()
    Y_tensor = tf.ragged.constant(dataset_complete_Y_list).to_tensor()
    
    # Save list
    open_file = open("/home/maxalaar/CodiumProjects/RFID_Deep_Learning/dataset/"+ str(name_save) +"_X.sav", "wb")
    pickle.dump(X_tensor, open_file)
    open_file.close()

    open_file = open("/home/maxalaar/CodiumProjects/RFID_Deep_Learning/dataset/"+ str(name_save) +"_Y.sav", "wb")
    pickle.dump(Y_tensor, open_file)
    open_file.close()

def load_tracking_dataset(name_dataset):
    print()
    print("--- Load Tracking Dataset ---")
    print()

    # Load list
    open_file = open("/home/maxalaar/CodiumProjects/RFID_Deep_Learning/dataset/"+ str(name_dataset) +"_X.sav", "rb")
    X_tensor = pickle.load(open_file)
    open_file.close()

    open_file = open("/home/maxalaar/CodiumProjects/RFID_Deep_Learning/dataset/"+ str(name_dataset) +"_Y.sav", "rb")
    Y_tensor = pickle.load(open_file)
    open_file.close()

    return X_tensor, Y_tensor