from general_include import *


def recurrent_neural_network_classifier_for_text_dataset(data_x=[], data_y=[], verbose=[], model = None):
    print()
    print("--- Recurrent Neural Network Classifier ---")
    print()

    # activation = 'tanh'
    activation = 'LeakyReLU'
    coeficient_dropout = 0.10
    batch_size = 128*2*2
    epochs_size = 600*2

    if model == None:
        # define the keras model
        model = Sequential()

        model.add(tf.keras.Input(shape=(len(data_x[0]), len(data_x[0][0]))))

        # model.add(tf.keras.layers.GaussianNoise(10))

        model.add(Dense(150, activation=activation))
        model.add(Dropout(coeficient_dropout))

        model.add(Dense(150, activation=activation))
        model.add(Dropout(coeficient_dropout))

        model.add(tf.keras.layers.SimpleRNN(100, return_sequences=True, activation=activation, recurrent_dropout=0, batch_input_shape=(batch_size, len(data_x[0]), len(data_x[0][0]))))

        model.add(Dense(80, activation=activation))
        model.add(Dropout(coeficient_dropout))

        model.add(Dense(80, activation=activation))
        model.add(Dropout(coeficient_dropout))

        model.add(tf.keras.layers.SimpleRNN(60, return_sequences=True, activation=activation, recurrent_dropout=0, batch_input_shape=(batch_size, len(data_x[0]), len(data_x[0][0]))))


        model.add(Dense(50, activation=activation))
        model.add(Dropout(coeficient_dropout))

        model.add(Dense(len(data_y[0][0]), activation='softmax'))
        model.add(Dropout(coeficient_dropout))
        
        # compile the keras model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy', f1_m])

    # fit the keras model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

    checkpoint_path = "./saved_model_nn/fit_checkpoint/training_RNN/cp-{epoch:04d}.ckpt"
    os.path.dirname(checkpoint_path)
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=5*batch_size)

    history = model.fit(data_x, data_y, epochs=epochs_size, batch_size=batch_size, validation_split=0.3, shuffle=True, verbose=1, callbacks=[es, cp]) 

    return model

def recurrent_neural_network_classifier_synt_tracking_dataset(data_x=[], data_y=[], verbose=[], model = None):
    print()
    print("--- Recurrent Neural Network Classifier ---")
    print()

    # activation = 'tanh'
    activation = 'LeakyReLU'
    coeficient_dropout = 0.06
    batch_size = 128*2*2
    epochs_size = 600*2

    if model == None:
        # define the keras model
        model = Sequential()

        model.add(tf.keras.Input(shape=(len(data_x[0]), len(data_x[0][0]))))

        # model.add(tf.keras.layers.GaussianNoise(15))

        # model.add(tf.keras.layers.SimpleRNN(150, return_sequences=True, activation=activation, recurrent_dropout=coeficient_dropout, batch_input_shape=(batch_size, len(data_x[0]), len(data_x[0][0]))))

        # model.add(Dense(150, activation=activation))
        # model.add(Dropout(coeficient_dropout))

        # model.add(Dense(150, activation=activation))
        # model.add(Dropout(coeficient_dropout))

        model.add(Dense(50, activation=activation))
        model.add(Dropout(coeficient_dropout))

        model.add(Dense(50, activation=activation))
        model.add(Dropout(coeficient_dropout))

        model.add(Dense(50, activation=activation))
        model.add(Dropout(coeficient_dropout))

        model.add(Dense(50, activation=activation))
        model.add(Dropout(coeficient_dropout))

        model.add(tf.keras.layers.SimpleRNN(60, return_sequences=True, activation=activation, recurrent_dropout=0, batch_input_shape=(batch_size, len(data_x[0]), len(data_x[0][0]))))

        model.add(Dense(50, activation=activation))
        model.add(Dropout(coeficient_dropout))

        model.add(Dense(len(data_y[0][0]), activation='softmax'))
        model.add(Dropout(coeficient_dropout))
        
        # compile the keras model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy', f1_m])

    # fit the keras model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

    checkpoint_path = "./saved_model_nn/fit_checkpoint/training_RNN/cp-{epoch:04d}.ckpt"
    os.path.dirname(checkpoint_path)
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=5*batch_size)

    history = model.fit(data_x, data_y, epochs=epochs_size, batch_size=batch_size, validation_split=0.3, shuffle=True, verbose=1, callbacks=[es, cp]) 

    return model