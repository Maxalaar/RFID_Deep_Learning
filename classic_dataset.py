from general_include import *

def load_classic_dataset(name_dataset, verbose=False):
  # Load the dataset
  dataset_data = arff.loadarff("./dataset/dataset_arff/"+str(name_dataset)+".arff")

  # Convert the dataset, arff to DataFram
  dataset_df = pd.DataFrame(dataset_data[0])

  if verbose == True:
    # Print the shape of data
    print("--- The shape of the "+str(name_dataset)+" dataset ---")
    print("The shape : ", dataset_df.shape)
    print()

    # Print info of the dataset
    print("--- Info of the "+str(name_dataset)+" dataset ---")
    dataset_df.info(verbose=True)
    print()

    # Print the 10 firsts lines of the dataset
    print("--- The firsts lines of the "+str(name_dataset)+" dataset ---")
    print(dataset_df.head(10))
    print()

  # Convert the dataset, DataFram to np
  dataset_np = dataset_df.to_numpy()

  # We divide the data-set between input and output
  dataset_np_X = dataset_np[:len(dataset_np), 0:int(len(dataset_np[0])-2)]
  dataset_np_Y = dataset_np[:len(dataset_np), int(len(dataset_np[0])-1)]

  # Convert object to int
  label_encoder = preprocessing.LabelEncoder()
  label_encoder.fit(dataset_np_Y)
  dataset_np_Y = label_encoder.transform(dataset_np_Y)

  return dataset_np_X, dataset_np_Y, label_encoder

def convert_classic_to_one_hot(data_x, data_y):
  # formatting the dataset
  data_x = np.asarray(data_x).astype('float32')
  data_y = np.asarray(data_y).astype('int')

  # Convert to one hot vector
  data_y_one_hot = np.zeros((data_y.size, data_y.max() + 1))
  data_y_one_hot[np.arange(data_y.size), data_y] = 1

  return data_x, data_y