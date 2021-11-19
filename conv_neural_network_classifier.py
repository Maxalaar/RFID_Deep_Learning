from general_include import *

def switch_data(data_x, data_y, column_1, column_2):
    column_1_val = data_x[column_1]
    column_2_val = data_x[column_2]

    data_x[column_1] = column_2_val
    data_x[column_2] = column_1_val

    column_1_val = data_y[column_1]
    column_2_val = data_y[column_2]

    data_y[column_1] = column_2_val
    data_y[column_2] = column_1_val

def neural_conv_network_classifier(data_x, data_y, verbose=[]):
    print("--- Neural Convolitional Network Classifier ---")
    print()

    # Number of available GPUs
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print()

    # Switch data
    # switch_data(data_x, data_y, 0, 1)

    # formatting the dataset
    data_x = np.asarray(data_x).astype('float32')
    data_x = data_x.reshape(data_x.shape[0], 1, data_x.shape[1])
    print(type(data_x))
    print(data_x.shape)

    data_y = np.asarray(data_y).astype('int')
    print(type(data_y))
    print(data_y.shape)

    # Convert to one hot vector
    data_y_one_hot = np.zeros((data_y.size, data_y.max() + 1))
    data_y_one_hot[np.arange(data_y.size), data_y] = 1

    data_y_one_hot = data_y_one_hot.reshape(data_y_one_hot.shape[0], 1, data_y_one_hot.shape[1])
    print(type(data_y_one_hot))
    print(data_y_one_hot.shape)

    # define the keras model
    model = Sequential()
    activation = 'LeakyReLU'

    # model.add(Dense(8, activation=activation))
    model.add(Conv1D(filters = 25, kernel_size = (5), activation=activation, padding="same"))
    model.add(Dropout(0.03))
    model.add(Dense(30, activation=activation))
    model.add(Dropout(0.03))
    model.add(Dense(len(data_y_one_hot[0][0]), activation='softmax'))

    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy'])

    # Fit the model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    history = model.fit(data_x, data_y_one_hot, epochs=int(1800*2), batch_size=128*2*2, shuffle=True, validation_split=0.3, verbose=1, callbacks=[es]) # , callbacks=[es]

    model.summary()

    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()