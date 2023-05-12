import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, GRU, RepeatVector, Dense, TimeDistributed, Input, BatchNormalization, \
    ConvLSTM2D, Flatten, Activation, Dot, concatenate, Bidirectional, Add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import numpy as np
import tsaug
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
import os
import random

keras = tf.keras

# ROCM
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/opt/rocm-5.3.0/lib'
os.environ['ROCM_PATH'] = '/opt/rocm'
# cuDNN
# os.environ['CUDA_HOME'] = '/apps/cuda/cuda-11.7.0'
# os.environ['CUDNN_HOME'] = '/apps/cuda/cudnn-8.5-cuda-11.7'
# os.environ['PATH'] = '{$CUDA_HOME}/bin:{$PATH}'
# os.environ['LD_LIBRARY_PATH'] = '{$CUDA_HOME}/lib64:{$CUDNN_HOME}/lib64:{$LD_LIBRARY_PATH}'


def lstm_fit(train, val, n_steps=128, features_out_num=1, features_out=range(1), features_in=range(8), fc_dim=50, lstm_dim_1=200, lstm_dim_2=200, batch_size=16, epochs=20, lr=0.0001):
    # Vanilla LSTM Model - single step output
    train_x, train_y = truncate_single_step(train, n_steps, features_in=features_in, features_out=features_out)
    val_x, val_y = truncate_single_step(val, n_steps, features_in=features_in, features_out=features_out)
    model = Sequential()
    model.add(LSTM(lstm_dim_1, activation='tanh', return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(LSTM(lstm_dim_2, activation='tanh', return_sequences=False))
    model.add(Dense(fc_dim, activation='elu'))
    model.add(Dense(features_out_num, activation='sigmoid'))
    model.summary()

    opt = keras.optimizers.Adam(learning_rate=lr)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=20, restore_best_weights=True)
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])
    history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=es)

    return model, history


def lstm_model(train, n_steps=128, features_out_num=1, features_out=range(1), features_in=range(8), fc_dim=16, lstm_dim_1=50, lstm_dim_2=100, batch_size=16, epochs=20, lr=0.0001):
    # Vanilla LSTM Model - single step output
    train_x, train_y = truncate_single_step(train, n_steps, features_in=features_in, features_out=features_out)
    model = Sequential()
    model.add(LSTM(lstm_dim_1, activation='tanh', return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(LSTM(lstm_dim_2, activation='tanh', return_sequences=False))
    model.add(Dense(fc_dim, activation='relu'))
    model.add(Dense(features_out_num, activation='sigmoid'))
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])

    return model


def bidirectional(train, n_steps=128, features_out_num=1, features_out=range(1), features_in=range(8), fc_dim=16, lstm_dim_1=128, lstm_dim_2=128, lstm_dim_3=32, batch_size=16, epochs=20, lr=0.0001):
    # Vanilla LSTM Model - single step output
    train_x, train_y = truncate_single_step(train, n_steps, features_in=features_in, features_out=features_out)

    model = Sequential()
    model.add(Input(shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Bidirectional(LSTM(lstm_dim_1, return_sequences=True)))
    model.add(LSTM(lstm_dim_2, activation='tanh', return_sequences=True))
    model.add(LSTM(lstm_dim_3, activation='tanh', return_sequences=False))
    model.add(Dense(fc_dim, activation='relu'))
    model.add(Dense(features_out_num, activation='sigmoid'))

    model.summary()
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])

    return model


def seq2seq_fit(train, val, n_steps=3, n_length=90, features_out_num=1, features_out=range(1), features_in=range(8), epochs=10, batch_size=50, lstm_dim=300, lstm_2_dim=300, fc_dim=20, lr=0.0001):
    # Sequence to sequence LSTM model - multi-step output
    # prepare data
    train_x, train_y = truncate_data(train, n_length*n_steps, n_length, features_in=features_in, features_out=features_out)
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    print(train_x.shape)
    # define model
    encoder_in = Input(shape=(n_length * n_steps, n_features))
    encoder = LSTM(lstm_dim, activation='tanh', return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c = encoder(encoder_in)
    # We discard `encoder_outputs` and only keep the states.
    # state_h = BatchNormalization(momentum=0.2)(state_h)
    # state_c = BatchNormalization(momentum=0.2)(state_c)
    encoder_states = [state_h, state_c]
    decoder_in = RepeatVector(train_y.shape[1])(state_h)
    decoder = LSTM(lstm_2_dim, activation='tanh', return_state=False,
                   return_sequences=True)(decoder_in, initial_state=encoder_states)
    
    # print(decoder.shape, encoder_outputs.shape)
    # attention = Dot(axes=[2, 2])([decoder, encoder_outputs])
    # attention = Activation('softmax')(attention)
    # context = Dot(axes=[2,1])([attention, encoder_outputs])
    # context = BatchNormalization(momentum=0.2)(context)
    # decoder_combined_context = concatenate([context, decoder])

    fc_layer = Dense(fc_dim, activation='elu')(decoder)
    out = TimeDistributed(Dense(features_out_num, activation='sigmoid'))(fc_layer)
    model = Model(inputs=encoder_in, outputs=out)

    es = EarlyStopping(monitor='val_loss', verbose=1, patience=30, restore_best_weights=True)
    # compile model
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, history


def seq2seq_compile(train, n_steps=3, n_length=90, features_out_num=1, features_out=range(1), features_in=range(8), lstm_dim=200, lstm_2_dim=200, fc_dim=20, lr=0.0001):
    # Compile only - no training
    # prepare data
    train_x, train_y = truncate_data(train, n_length*n_steps, n_length, features_in=features_in, features_out=features_out)
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    print(train_x.shape)
    # define model
    encoder_in = Input(shape=(n_length * n_steps, n_features))
    encoder = LSTM(lstm_dim, dropout=0.2, recurrent_dropout=0.2, activation='tanh', return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_in)
    # We discard `encoder_outputs` and only keep the states.
    state_h = BatchNormalization(momentum=0.3)(state_h)
    state_c = BatchNormalization(momentum=0.3)(state_c)
    encoder_states = [state_h, state_c]
    decoder = RepeatVector(train_y.shape[1])(state_h)
    decoder = LSTM(lstm_2_dim, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_state=False,
                   return_sequences=True)(decoder, initial_state=encoder_states)
    fc_layer = TimeDistributed(Dense(fc_dim, activation='relu'))(decoder)
    out = TimeDistributed(Dense(features_out_num, activation='sigmoid'))(fc_layer)
    model = Model(inputs=encoder_in, outputs=out)

    # compile model
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])

    return model


# convert history into inputs and outputs
def truncate_data(data, n_input=150 * 5, n_out=150, features_in=range(8), features_out=range(1)):
    X, y = [], []
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, features_in])
            y.append(data[in_end:out_end, features_out])
            # move along one time step
        in_start += 1
    return np.array(X), np.array(y)



# convert history into inputs and outputs
def truncate_single_step(data, n_steps=128, features_in=range(8), features_out=range(1)):
    X, y = [], []
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_steps
        # ensure we have enough data for this instance
        if in_end < len(data):
            X.append(data[in_start:in_end, features_in])
            y.append(data[in_end, features_out])
            # move along one time step
        in_start += 1
    data_x = np.array(X)
    # data_x = data_x.reshape((data_x.shape[0], data_x.shape[1], len(features_in)))
    data_y = np.array(y)
    # data_y = data_y.reshape((data_y.shape[0], data_y.shape[1], len(features_out)))
    return data_x, data_y


def evaluate_forecasts(actual, predicted):
    scores = []
    # calculate an RMSE score for time step
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = np.sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = np.sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


def evaluate_model(model, train, test, n_steps, n_length, features_in_num=8, features=range(8), features_out_num=5, features_out=range(5), batch_size=50, initial_epochs=5, epochs=1, lr=0.0001, batches_to_train=8):
    # history is the existing data
    print(train.shape, test.shape)
    tr_rem = train.shape[0] % n_length
    if tr_rem != 0:
        train = train[tr_rem:]
    train = train[:, features]
    history = train.reshape(int(train.shape[0] / n_length), n_length, features_in_num)
    history = history[:, :, features]
    # walk-forward validation over each fixed interval
    ts_rem = test.shape[0] % n_length
    if ts_rem != 0:
        test = test[:-ts_rem]
    predictions = []
    test = test[:, features_out]
    test_arr = test.reshape(int(test.shape[0] / n_length), n_length, features_in_num)
    print(test_arr.shape, history.shape)

    # Transfer learning training of initial history
    # x = model.get_layer('time_distributed').output
    # out = TimeDistributed(Dense(features_out_num, activation='sigmoid'), name='td_dense_out')(x)
    # model = keras.models.Model(inputs=model.input, outputs=out)
    # opt = keras.optimizers.Adam(learning_rate=lr)
    # model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])
    # train_hist = history.reshape((n_length * history.shape[0], features_in_num))
    # train_x, train_y = truncate_data(train_hist, n_steps * n_length, n_length, features_out=features_out)
    # model.fit(train_x, train_y, epochs=initial_epochs, batch_size=batch_size, verbose=1)
    # model.reset_states()

    for i in range(test_arr.shape[0]):
        # predict the next odds 
        yhat_sequence = make_forecast(model, history, n_steps, n_length)
        # store the predictions
        predictions.append(yhat_sequence)
        test_node = test_arr[i, :, :].reshape(1, test_arr.shape[1], test_arr.shape[2])
        # get real observation and add to history for predicting the next batch
        history = np.append(history, test_node, axis=0)
        # evaluate predictions days for each batch every x batches
        # if i % batches_to_train == 0:
        #     train_hist = history.reshape((n_length * history.shape[0], features_in_num))
        #     train_x, train_y = truncate_data(train_hist, n_steps * n_length, n_length, features_out=features_out)
        #     model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
        #     model.reset_states()

    predictions = np.array(predictions)
    predictions = predictions.squeeze()
    actual = test_arr
    if predictions.ndim == 3:
        predictions = predictions[:, :, 0]
    if actual.ndim == 3:
        actual = test_arr[:, :, 0]
    print(actual.shape, predictions.shape)
    score, scores = evaluate_forecasts(actual, predictions)
    return score, scores, actual, predictions


def make_forecast(model, history, n_steps, n_length):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input = data[-(n_steps * n_length):, :]
    input_x = input.reshape((1, input.shape[0], input.shape[1]))
    # reshape into [samples, time steps, rows, cols, channels]
    # forecast the batch
    yhat = model.predict(input_x, verbose=1)
    # we only want the vector forecast
    yhat = yhat[0]
    # y = np.concatenate((input[:, 0], yhat[:,0]))
    # plt.plot(y)
    return yhat