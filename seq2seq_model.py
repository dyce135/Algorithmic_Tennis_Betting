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


def truncate(x, features=7, x_len=750, y_len=150):
    input, output = [], []
    feature_cols, target_cols = range(features), range(features)
    for i in range(len(x) - x_len - y_len + 1):
        input.append(x[i:(i + x_len), feature_cols].tolist())
        output.append(x[(i + x_len):(i + x_len + y_len), target_cols].tolist())

    return np.array(input), np.array(output)


def seq2seq_fit(train, n_steps=3, n_length=90, features=7, features_list=range(1), epochs=10, batch_size=50, lstm_dim=200, lstm_2_dim=200, fc_dim=10, lr=0.0001):
    # prepare data
    train_x, train_y = truncate_data(train, n_length*n_steps, n_length, features=features_list)
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    # define model
    model = Sequential()
    model.add(LSTM(lstm_dim, dropout=0.2, recurrent_dropout=0.2, activation='elu',
                         input_shape=(n_length * n_steps, n_features)))
    model.add(BatchNormalization(momentum=0.4))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(lstm_2_dim, dropout=0.2, recurrent_dropout=0.2, activation='elu', return_sequences=True))
    model.add(TimeDistributed(Dense(fc_dim, activation='relu')))
    model.add(TimeDistributed(Dense(features, activation='sigmoid')))

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, history


# convert history into inputs and outputs
def truncate_data(data, n_input=150 * 5, n_out=150, features=range(7)):
    X, y = [], []
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, [0, 5, 6]])
            y.append(data[in_end:out_end, features])
            # move along one time step
        in_start += 1
    return np.array(X), np.array(y)


def evaluate_forecasts(actual, predicted):
    scores = list()
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


def evaluate_model(model, train, test, n_steps, n_length, n_input):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = make_forecast(model, history, n_steps, n_length, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
        # evaluate predictions days for each week
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


def make_forecast(model, history, n_steps=3, n_length=90):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    print(data.shape)
    # retrieve last observations for input data
    input = data[-(501 * n_steps * n_length):-(500 * n_steps * n_length), :]
    input_x = input.reshape((1, input.shape[0], input.shape[1]))
    print(input.shape)
    # reshape into [samples, time steps, rows, cols, channels]
    # forecast the next week
    yhat = model.predict(input_x, verbose=1)
    # we only want the vector forecast
    yhat = yhat[0]
    y = np.concatenate((input[:, 0], yhat[:,0]))
    plt.plot(y)
    return yhat