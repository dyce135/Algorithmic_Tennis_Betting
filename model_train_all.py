import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import mean_absolute_error as mae
import seq2seq_model
from tensorflow.keras.models import load_model, Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, TimeDistributed, Add, Bidirectional, LSTM, Flatten, Dropout
from matplotlib import pyplot as plt
import os
from os.path import join
from random import shuffle

train_list = os.listdir('./Data/Train')
shuffle(train_list)

n_length = 75
n_steps = 1
features_in = range(1)
features_out = range(1)
features_out_num = 1
epochs = 10
batch_size = 15
lr = 0.0001

df = pd.read_csv(join('./Data/Train', train_list[0]), index_col=0)
data = df.to_numpy()
# initiate untrained model
model = seq2seq_model.lstm_model(data, n_steps=n_length, features_out_num=features_out_num, lstm_dim_1=200, lstm_dim_2=200,
                                 features_in=features_in, features_out=features_out, lr=lr)

# fit all training data
for file in train_list:
    print('Fitting ', file)

    # read data
    df = pd.read_csv(join('./Data/Train', file), index_col=0)
    data = df.to_numpy()

    train_x, train_y = seq2seq_model.truncate_single_step(data, n_steps=n_length, features_in=features_in,
                                                          features_out=features_out)

    print(train_y.shape, train_x.shape)
    # replace last two fully connected layers
    x = model.get_layer('lstm_1').output
    x = Dropout(0.2)(x)
    x = Dense(16, activation='elu')(x)
    x = Dense(features_out_num, activation='sigmoid')(x)
    model = keras.models.Model(inputs=model.input, outputs=x)

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    hist_df = pd.DataFrame(history.history)
    # reset lstm states
    model.reset_states()

    # augmented time series training
    # if only odds as feature
    # if features_out_num == 1:
    #     # create flipped data
    #     data[:, 0] = - data[:, 0] + 1
    #     train_x_flipped, train_y_flipped = seq2seq_model.truncate_single_step(data, n_steps=n_length, features_in=features_in, features_out=features_out)
    #     x = model.get_layer('lstm_1').output
    #     x = Dense(50, activation='elu')(x)
    #     x = Dense(features_out_num, activation='sigmoid')(x)
    #     model = keras.models.Model(inputs=model.input, outputs=x)
    #     opt = keras.optimizers.Adam(learning_rate=lr)
    #     model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])
    #     # print(model.get_layer('lstm').get_weights())
    #     history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
    #     hist_df = pd.DataFrame(history.history)
    #     model.reset_states()

print('Saving model.')
model.save("transfer_lstm_model", save_format='tf')
print('Model saved to /transfer_lstm_model.')
