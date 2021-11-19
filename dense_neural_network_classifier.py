from general_include import *

def prerceptron_classifier(data_x, data_y, verbose=[]):
    print("--- Prerceptron Classifier ---")
    print()

    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(data_x, data_y)
    print(clf.score(data_x, data_y))

def dense_neural_network_classifier(data_x, data_y, verbose=[]):
    print("--- Neural Network Classifier ---")
    print()

    # Number of available GPUs
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print()

    data_y_one_hot = data_y

    # define the keras model
    model = Sequential()
    activation = 'LeakyReLU'
    coeficient_dropout = 0.04

    model.add(Dense(50, activation=activation))
    model.add(Dropout(coeficient_dropout))

    model.add(Dense(50, activation=activation))
    model.add(Dropout(coeficient_dropout))

    model.add(Dense(50, activation=activation))
    model.add(Dropout(coeficient_dropout))

    model.add(Dense(50, activation=activation))
    model.add(Dropout(coeficient_dropout))

    model.add(Dense(50, activation=activation))
    model.add(Dropout(coeficient_dropout))

    model.add(Dense(len(data_y_one_hot[0]), activation='softmax'))
    model.add(Dropout(coeficient_dropout))

    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy', f1_m])

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