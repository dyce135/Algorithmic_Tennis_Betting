import numpy as np
import pandas as pd
import json
from datetime import datetime
from dateutil import parser


def get_info(file):
    with open(file, 'r') as f:
        top = f.readline()
        top = json.loads(top)
    market_datetime = top['mc'][0]['marketDefinition']['marketTime']
    eventId = top['mc'][0]['marketDefinition']['eventId']
    runners = top['mc'][0]['marketDefinition']['runners']
    print("Event ID: " + eventId)
    print("Market Time: " + market_datetime)
    print("Runner IDs and names: \n" + str(runners[0]['id']) + ", " + runners[0]['name'] + "\n" + str(runners[1]['id'])
          + ", " + runners[1]['name'])

def get_list(file, runner_id):
    data_list = []
    for line in open(file, 'r'):
        data_list.append(json.loads(line))
    market_datetime = parser.parse(data_list[-1]['mc'][0]['marketDefinition']['marketTime'])
    market_timestamp = datetime.timestamp(market_datetime) * 1000
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
    implied_odds = np.array([1 / arr[:, 0], arr[:, 1]]).T

    return implied_odds


# Find avg ltp odds
def odds_avg(file, runner_id_1, runner_id_2):
    runner_list_1 = get_list(file, runner_id_1)
    runner_list_2 = get_list(file, runner_id_2)
    runner_1 = convert_odds(runner_list_1)
    runner_2 = convert_odds(runner_list_2)
    if runner_1[-1, 1] > runner_2[-1, 1]:
        if runner_1[1, 1] > runner_2[1, 1]:
            timestamps = np.arange(round(runner_2[1, 1], -2), round(runner_1[-1, 1] + 1, -2), 100)
        else:
            timestamps = np.arange(round(runner_1[1, 1], -2), round(runner_1[-1, 1] + 1, -2), 100)
    else:
        if runner_1[1, 1] > runner_2[1, 1]:
            timestamps = np.arange(round(runner_2[1, 1], -2), round(runner_2[-1, 1] + 1, -2), 100)
        else:
            timestamps = np.arange(round(runner_1[1, 1], -2), round(runner_2[-1, 1] + 1, -2), 100)

    odds = np.zeros(np.shape(timestamps))
    df_timestamps = pd.Series(timestamps)
    df_datetime = pd.to_datetime(df_timestamps, unit='ms')

    df = pd.DataFrame({'runner 1': odds, '1 - runner 2': odds}, index=df_datetime)

    for index, time in enumerate(runner_1[:, 1]):
        df['runner 1'].loc[pd.to_datetime(round(time, -2), unit='ms')] = runner_1[index, 0]

    for index, time in enumerate(runner_2[:, 1]):
        df['1 - runner 2'].loc[pd.to_datetime(round(time, -2), unit='ms')] = 1 - runner_2[index, 0]

    df.replace(0, np.nan, inplace=True)
    df.interpolate(method='time', limit_direction='both', inplace=True)
    df['avg'] = df.mean(axis=1)
    df = df.resample('250ms').last()
    print(df)
    return df


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
