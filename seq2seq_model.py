import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, GRU, RepeatVector, Dense, TimeDistributed, Input, BatchNormalization, \
    ConvLSTM2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import numpy as np
import os

keras = tf.keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/opt/rocm-5.3.0/lib'
os.environ['ROCM_PATH'] = '/opt/rocm'


def seq2seq_fit(train, n_steps=3, n_length=90, features_out_num=1, features_out=range(1), features_in=range(8), epochs=10, batch_size=50, lstm_dim=200, lstm_2_dim=200, fc_dim=10, lr=0.0001):
    # prepare data
    train_x, train_y = truncate_data(train, n_length*n_steps, n_length, features_in=features_in, features_out=features_out)
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    # define model
    encoder_in = Input(shape=(n_length * n_steps, n_features))
    encoder = LSTM(lstm_dim, dropout=0.2, recurrent_dropout=0.2, activation='tanh', return_state=True)
    state_h, encoder_outputs, state_c = encoder(encoder_in)
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
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, history


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


def evaluate_forecasts(actual, predicted):
    scores = []
    # calculate an RMSE score for each day
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


def evaluate_model(model, train, test, n_steps, n_length, features=range(8)):
    # history is a list of weekly data
    print(train.shape, test.shape)
    tr_rem = train.shape[0] % n_length
    if tr_rem != 0:
        train = train[tr_rem:]
    history = train.reshape(int(train.shape[0] / n_length), n_length, 8)
    history = history[:, :, features]
    # walk-forward validation over each week
    ts_rem = test.shape[0] % n_length
    if ts_rem != 0:
        test = test[:-ts_rem]
    predictions = []
    test_arr = test.reshape(int(test.shape[0] / n_length), n_length, 8)
    test_arr = test_arr[:, :, features]
    print(test_arr.shape, history.shape)
    for i in range(test_arr.shape[0]):
        # predict the week
        yhat_sequence = make_forecast(model, history, n_steps, n_length)
        # store the predictions
        predictions.append(yhat_sequence)
        test_node = test_arr[i, :, :].reshape(1, test_arr.shape[1], test_arr.shape[2])
        print(test_node.shape, history.shape)
        # get real observation and add to history for predicting the next week
        history = np.append(history, test_node, axis=0)
        # evaluate predictions days for each week
        # model.fit()
    predictions = np.array(predictions)
    predictions = predictions.squeeze()
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
    # forecast the next week
    yhat = model.predict(input_x, verbose=1)
    # we only want the vector forecast
    yhat = yhat[0]
    y = np.concatenate((input[:, 0], yhat[:,0]))
    # plt.plot(y)
    return yhat