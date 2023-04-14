import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import markov_sim as markov
import json
import os
from datetime import datetime
from re import sub
from dateutil import  parser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/opt/rocm-5.3.0/lib'

data_list = []

for line in open('1.145414064.json', 'r'):
    data_list.append(json.loads(line))

runner_id = 2249834
runner_id_2 = 2251410
market_datetime = parser.parse(data_list[0]['mc'][0]['marketDefinition']['marketTime'])
market_timestamp = datetime.timestamp(market_datetime) * 1000


def get_list(runner_id):
    # Create list for each runner
    runner_list = []
    for instance in data_list:
        if instance['pt'] > market_timestamp:
            if instance['mc'][0]['rc']:
                # Check for runner id
                temp_dict = {k: v for (k, v) in instance['mc'][0]['rc'][0].items() if v == runner_id}
                if temp_dict:
                    # Append runner info
                    runner_list.append([instance['mc'][0]['rc'][0], instance['pt']])
                elif len(instance['mc'][0]['rc']) > 1:
                    # If more than one runner
                    temp_dict_2 = {k: v for (k, v) in instance['mc'][0]['rc'][1].items() if v == runner_id}
                    if temp_dict_2:
                        runner_list.append([instance['mc'][0]['rc'][1], instance['pt']])
    return runner_list


def convert_odds(runner_list):
    # Convert to back/lay/last traded odds

    list = []

    for item in runner_list:
        if 'ltp' in item[0]:
            list.append([item[0]['ltp'], item[1]])

    del list[-1]
    arr = np.array(list)
    arr = arr[arr[:, 0] != 0]
    arr[:, 1] = arr[:, 1] - market_timestamp
    implied_odds = np.array([1 / arr[:, 0], arr[:, 1]]).T

    return implied_odds


# Find avg ltp odds
def odds_avg(runner_1, runner_2):
    if runner_1[-1, 1] > runner_2[-1, 1]:
        shape = round(runner_1[-1, 1] / 100)
    else:
        shape = round(runner_2[-1, 1] / 100)

    odds_1 = np.empty(shape)
    odds_2 = np.empty(shape)

    for element in runner_1:
        odds_1[round(element[1] / 100) - 1] = element[0]

    for element in runner_2:
        odds_2[round(element[1] / 100) - 1] = element[0]

    df = pd.DataFrame({'runner 1': odds_1, 'runner_2': odds_2})
    print(df.head())
    df.replace(0, np.nan, inplace=True)
    df.interpolate(method='linear')
    df['avg'] = df.mean(axis=1)
    print(df.head())
    avg = df['avg'].to_numpy()

    return avg

def calc_pup():
    pass


def get_current_spread():
    pass


def calc_future_spread():
    pass


def calc_score():
    pass


def get_markov_odds():
    pass


runner_list_1 = get_list(runner_id)
runner_list_2 = get_list(runner_id_2)
runner_odds_1 = convert_odds(runner_list_1)
runner_odds_2 = convert_odds(runner_list_2)
avg_odds = odds_avg(runner_odds_1, runner_odds_2)

plt.plot(runner_odds_1[:, 1], runner_odds_1[:, 0])
plt.show()
plt.plot(runner_odds_2[:, 1], runner_odds_2[:, 0])
plt.show()

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
