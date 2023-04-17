import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/opt/rocm-5.3.0/lib'

with open('1.209134319.json') as data_file:
    data_dict = json.load(data_file)

data_dict = data_dict['mcm']

runner_list = []
runner_id = 8859306
runner_id_2 = 2249229

for item in data_dict.items():
    if 'rc' in item[1]:
        temp_dict = {k: v for (k, v) in item[1]['rc'][0].items() if v == runner_id}
        if temp_dict:
            runner_list.append(item[1]['rc'][0])
        elif len(item[1]['rc']) > 1:
            temp_dict_2 = {k: v for (k, v) in item[1]['rc'][1].items() if v == runner_id}
            if temp_dict_2:
                    runner_list.append(item[1]['rc'][1])

back_list = []

for item in runner_list:
    if 'batb' in item:
        back_list.append(item['batb'][0][1])
    elif 'bdatb' in item:
        back_list.append(item['bdatb'][0][1])

back_arr = np.array(back_list)
back_arr = back_arr[back_arr != 0]
plt.plot(back_arr)
plt.show()
implied_odds = 1 / back_arr

# training_data_len = math.ceil(len(implied_odds) * 0.8)
#
# values = implied_odds.reshape(-1, 1)
# train_data = values[0: training_data_len, :]
# x_train = []
# y_train = []
#
# for i in range(60, len(train_data)):
#     x_train.append(train_data[i - 60:i, 0])
#     y_train.append(train_data[i, 0])
#
# x_train, y_train = np.array(x_train), np.array(y_train)
#
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#
# test_data = values[training_data_len-60: , : ]
# x_test = []
# y_test = values[training_data_len:]
#
# for i in range(60, len(test_data)):
#     x_test.append(test_data[i-60:i, 0])
#
# x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# #%%
# model = keras.Sequential()
# model.add(layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(layers.LSTM(64, return_sequences=False))
# model.add(layers.Dense(16))
# model.add(layers.Dense(1))
# model.summary()
#
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, batch_size= 10, epochs=20)