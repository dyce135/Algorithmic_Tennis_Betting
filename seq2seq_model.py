import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, GRU, RepeatVector, Dense, TimeDistributed, Input, BatchNormalization, \
    ConvLSTM2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
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


def convlstm_seq2seq_fit(train, n_steps=5, n_length=150, features=7, epochs=10, batch_size=128, filters=64, lstm_dim=200, fc_dim=100, lr=0.0001):
    # prepare data
    train_x, train_y = to_supervised(train, n_length*n_steps, n_length, n_features=1)
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape into subsequences [samples, time steps, rows, cols, channels]
    train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], train_y.shape[2]))

    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=filters, dropout=0.2, recurrent_dropout=0.2, kernel_size=(1, 3), activation='elu',
                         input_shape=(n_steps, 1, n_length, n_features)))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(lstm_dim, dropout=0.2, recurrent_dropout=0.2, activation='elu', return_sequences=True))
    model.add(TimeDistributed(Dense(fc_dim, activation='relu')))
    model.add(TimeDistributed(Dense(features, activation='sigmoid')))

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, history


# def lstm_seq2seq_fit(train, n_steps=5, n_length=150, features=7, epochs=10, batch_size=128, filters=64, lstm_dim=200, fc_dim=100, lr=0.001):
#     # prepare data
#     train_x, train_y = to_supervised(train, n_length*n_steps, n_length, features)
#     # define parameters
#     n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
#     # reshape output into [samples, timesteps, features]
#     train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], features))
#     # define model
#     model = Sequential()
#     model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, activation='elu', input_shape=(n_timesteps, n_features)))
#     model.add(BatchNormalization(momentum=0.3))
#     model.add(RepeatVector(n_outputs))
#     model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, activation='elu', return_sequences=True))
#     model.add(TimeDistributed(Dense(32, activation='relu')))
#     model.add(TimeDistributed(Dense(1, activation='sigmoid')))
#     model.compile(loss='mse', optimizer='adam')
#     # fit network
#     history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
#     return model, history


# convert history into inputs and outputs
def to_supervised(data, n_input=150 * 5, n_out=150, n_features=7):
    X, y = [], []
    in_start = 0
    features = range(n_features)
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, features])
            # move along one time step
        in_start += 1
    return np.array(X), np.array(y)