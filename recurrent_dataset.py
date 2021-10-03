from general_include import *

def create_tracking_dataset(num_colonne_room, num_row_room, name_dataset, name_save, num_element_dataset, num_element_path=30, num_repetitions_zone=1, verbose=False):
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
    num_colonne = num_colonne_room # 10 # 14 for all room
    num_row = num_row_room # 19

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
            
            if verbose:
                print(str(alphabet[i] + str(j+1)) + " : " + str(i) + ", " + str(j))
    
    if verbose:
        print()

    list_list_zone_index = []
    for _ in range (0, num_element_dataset):

        list_zone_index = []

        # We choose a init zone and target zone at random for creat list of zone
        x_zone_init = random.randrange(0, num_colonne, 1)
        y_zone_init = random.randrange(0, num_row, 1)

        x_zone = x_zone_init
        y_zone = y_zone_init
        
        # if isinstance(num_repetitions_target, int):
        #     num_repetitions_target_prov = num_repetitions_target
        # else:
        #     num_repetitions_target_prov = random.randint(num_repetitions_target[0], num_repetitions_target[1])

        frame = []
        data = None
        


        # add init zone in zone list
        if isinstance(num_repetitions_zone, int):
            num_repetitions_zone_prov = num_repetitions_zone
        else:
            num_repetitions_zone_prov = random.randint(num_repetitions_zone[0], num_repetitions_zone[1])
            
        for j in range(0, num_repetitions_zone_prov):
            if len(list_zone_index) < num_element_path:
                list_zone_index.append((x_zone, y_zone))

        # for i in range(0, num_repetitions_target_prov):
        while len(list_zone_index) < num_element_path:

            x_zone_target = random.randrange(0, num_colonne, 1)
            y_zone_target = random.randrange(0, num_row, 1)

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
                if isinstance(num_repetitions_zone, int):
                    num_repetitions_zone_prov = num_repetitions_zone
                else:
                    num_repetitions_zone_prov = random.randint(num_repetitions_zone[0], num_repetitions_zone[1])
                    
                for j in range(0, num_repetitions_zone_prov):
                    if len(list_zone_index) < num_element_path:
                        list_zone_index.append((x_zone, y_zone))

        # for each zone give some measure
        for i in range(0, len(list_zone_index)):
            frame.append(map_zone[list_zone_index[i][0]][list_zone_index[i][1]].sample())

        # each frame is compile in data
        list_list_zone_index.append(list_zone_index)
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
        dataset_complete_X_list.append(dataset_complete_X_pandas[i].to_numpy())
        dataset_complete_Y_list.append(dataset_complete_Y_pandas[i].to_numpy())

    # Convert Python list to Tensorflow Tensor
    X_tensor = tf.convert_to_tensor(dataset_complete_X_list)
    Y_tensor = tf.convert_to_tensor(dataset_complete_Y_list)

    # X_tensor = tf.ragged.constant(dataset_complete_X_list)
    # Y_tensor = tf.ragged.constant(dataset_complete_Y_list)

    # print(type(X_tensor))
    # X_tensor = X_tensor.to_tensor()
    # Y_tensor = Y_tensor.to_tensor()

    if verbose:
        for i in range(0, len(X_tensor)):
            print("- For path " + str(i) + " -")

            # Print the path itinerary
            print("List index zones : ")
            for j in range(0, len(list_list_zone_index[i])):
                print(str(alphabet[list_list_zone_index[i][j][0]]), end = '')
                print(str(list_list_zone_index[i][j][1]+1), end = '')
                print(" " + str(list_list_zone_index[i][j]) + "; ", end = '')
            print()
            print()

            print("X:")
            print(X_tensor[i])
            print("Y:")
            print(Y_tensor[i])
            print()
    
    # Save tensor
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