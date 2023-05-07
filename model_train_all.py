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
from tensorflow.keras.layers import Dense, TimeDistributed
from matplotlib import pyplot as plt
import os
from os.path import join
from random import shuffle

train_list = os.listdir('./Data/Train')
shuffle(train_list)

n_length = 36
n_steps = 1
features_in = range(14) 
features_out = range(14)
features_out_num = 14
epochs=10
batch_size=50
lr = 0.0001

df = pd.read_csv(join('./Data/Train', train_list[0]), index_col=0)
data = df.to_numpy()
model = seq2seq_model.lstm_compile(data, n_steps=n_length, features_out_num=features_out_num, 
                                   features_in=features_in, features_out=features_out, lr=lr)

for file in train_list:
    print('Fitting ', file)

    df = pd.read_csv(join('./Data/Train', file), index_col=0)
    data = df.to_numpy()
    
    train_x, train_y = seq2seq_model.truncate_single_step(data, n_steps=n_length, features_in=features_in, features_out=features_out)
    x = model.get_layer('lstm_2').output
    x = Dense(features_out_num, activation='sigmoid')(x)
    model = keras.models.Model(inputs=model.input, outputs=x)
    
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])
    # print(model.get_layer('lstm').get_weights())
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(join('Train_hist/', file))
    model.reset_states()

print('Saving model.')
model.save("seq2seq_model", save_format='tf')
print('Model saved to /seq2seq_model.')
