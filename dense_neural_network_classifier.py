from general_include import *

def dense_neural_network_classifier(data_x, data_y, verbose=[]):
    print("--- Neural Network Classifier ---")
    print()

    # Number of available GPUs
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print()

    # formatting the dataset
    data_x = np.asarray(data_x).astype('float32')
    data_y = np.asarray(data_y).astype('int')

    # Convert to one hot vector
    data_y_one_hot = np.zeros((data_y.size, data_y.max() + 1))
    data_y_one_hot[np.arange(data_y.size), data_y] = 1

    # define the keras model
    model = Sequential()
    activation = 'LeakyReLU'

    model.add(Dense(50, activation=activation))
    model.add(Dense(50, activation=activation))
    model.add(Dense(50, activation=activation))
    model.add(Dense(50, activation=activation))
    model.add(Dense(50, activation=activation))
    
    # nbr_neurone_mid_layer = 50
    # dropout_mid_layer = 0.08

    # nbr_layer = 2
    # coef_neurone = 1
    # coef_dropout = 1

    # for i in range(nbr_layer):
    #   neurone_prov = int(nbr_neurone_mid_layer * pow(coef_neurone, (nbr_layer - (i))))
    #   model.add(Dense(neurone_prov, activation=activation))

    #   if i != 0:
    #     dropout_prov = dropout_mid_layer * pow(coef_dropout, (nbr_layer - (i)))
    #     model.add(Dropout(dropout_prov))
      
    # model.add(Dense(nbr_neurone_mid_layer, activation=activation))
    # model.add(Dropout(dropout_mid_layer))

    # for i in range(nbr_layer):
    #   neurone_prov = int(nbr_neurone_mid_layer * pow(coef_neurone, i+1))
    #   model.add(Dense(neurone_prov, activation=activation))

    #   if i != nbr_layer-1:
    #     dropout_prov = dropout_mid_layer * pow(coef_dropout, i+1)
    #     model.add(Dropout(dropout_prov))

    model.add(Dense(len(data_y_one_hot[0]), activation='softmax'))

    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy'])

    # Fit the model
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    history = model.fit(data_x, data_y_one_hot, epochs=1800*2, batch_size=128*2*2, shuffle=True, validation_split=0.3, verbose=1) # , callbacks=[es]
    # history = model.fit(data_x, data_y_one_hot, epochs=1, batch_size=128*2*2, shuffle=True, validation_split=0.3, verbose=1) # , callbacks=[es]

    model.summary()

    return model

    # list all data in history
    # print(history.history.keys())

    # # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()