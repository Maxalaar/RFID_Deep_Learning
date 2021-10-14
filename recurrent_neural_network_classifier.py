from general_include import *


def recurrent_neural_network_classifier(data_x=[], data_y=[], verbose=[], model = None):
    print()
    print("--- Recurrent Neural Network Classifier ---")
    print()

    # activation = 'tanh'
    activation = 'LeakyReLU'
    coeficient_dropout = 0.0
    batch_size = 128*2*2
    epochs_size = 600

    if model == None:
        # define the keras model
        model = Sequential()

        model.add(tf.keras.Input(shape=(len(data_x[0]), len(data_x[0][0]))))

        # model.add(tf.keras.layers.GaussianNoise(30))
        model.add(tf.keras.layers.SimpleRNN(50, return_sequences=True, activation=activation, recurrent_dropout=0.03, batch_input_shape=(batch_size, len(data_x[0]), len(data_x[0][0]))))

        model.add(Dense(50, activation=activation))
        model.add(Dropout(0))
        # model.add(Dense(50, activation=activation))
        # model.add(Dropout(coeficient_dropout))
        # model.add(Dense(50, activation=activation))
        # model.add(Dropout(coeficient_dropout))
        # model.add(Dense(50, activation=activation))
        # model.add(Dropout(coeficient_dropout))
        # model.add(Dense(50, activation=activation))
        # model.add(Dropout(coeficient_dropout))

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
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy', f1_m])

    # fit the keras model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

    checkpoint_path = "./saved_model_nn/fit_checkpoint/training_RNN/cp-{epoch:04d}.ckpt"
    os.path.dirname(checkpoint_path)
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=5*batch_size)

    history = model.fit(data_x, data_y, epochs=epochs_size, batch_size=batch_size, validation_split=0.3, shuffle=True, verbose=1, callbacks=[es, cp]) 

    return model