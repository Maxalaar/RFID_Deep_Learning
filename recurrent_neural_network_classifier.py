from general_include import *

def recurrent_neural_network_classifier(data_x, data_y, verbose=[]):
    print()
    print("--- Recurrent Neural Network Classifier ---")
    print()

    # define the keras model
    model = Sequential()
    # activation = 'tanh'
    activation = 'LeakyReLU'
    batch_size = 128*2*2
    coeficient_dropout = 0.04

    model.add(tf.keras.Input(shape=(len(data_x[0]), len(data_x[0][0]))))

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

    model.add(tf.keras.layers.SimpleRNN(50, return_sequences=True, activation=activation, recurrent_dropout=coeficient_dropout, batch_input_shape=(batch_size, len(data_x[0]), len(data_x[0][0]))))

    # number_layers = 5
    # number_neurons = 50
    # coeficient_dropout = 0.10

    # for _ in range(0, number_layers):
    #     model.add(Dense(number_neurons, 
    #                 activation=activation,
    #                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    #                 bias_regularizer=regularizers.l2(1e-4),
    #                 activity_regularizer=regularizers.l2(1e-5)))
    #     model.add(Dropout(coeficient_dropout))

    # # for _ in range(0, number_layers):
    # #     model.add(Dense(number_neurons, activation=activation))
    # #     model.add(Dropout(coeficient_dropout))

    # model.add(tf.keras.layers.SimpleRNN(60, return_sequences=True, activation=activation, batch_input_shape=(batch_size, len(data_x[0]), len(data_x[0][0]))))
    # model.add(tf.keras.layers.SimpleRNN(70, return_sequences=True, activation=activation, batch_input_shape=(batch_size, len(data_x[0]), len(data_x[0][0]))))

    model.add(Dense(len(data_y[0][0]), activation='softmax'))
    
    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy'])

    # Fit the model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    history = model.fit(data_x, data_y, epochs=int(1800*2), batch_size=batch_size, validation_split=0.3, shuffle=True, verbose=1, callbacks=[es])