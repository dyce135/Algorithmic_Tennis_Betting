import numpy as np
import pandas as pd
import json
from datetime import datetime
from dateutil import parser
from hampel import hampel


def get_data(file):
    data_list = []
    for line in open(file, 'r'):
        data_list.append(json.loads(line))

    with open(file, 'r') as f:
        top = f.readline()
        top = json.loads(top)

    runner_id = top['mc'][0]['marketDefinition']['runners'][0]['id']
    runner_id_2 = top['mc'][0]['marketDefinition']['runners'][1]['id']
    runner_name = top['mc'][0]['marketDefinition']['runners'][0]['name']
    runner_name_2 = top['mc'][0]['marketDefinition']['runners'][1]['name']
    market_datetime = parser.parse(data_list[-1]['mc'][0]['marketDefinition']['marketTime'])
    r1_result = data_list[-1]['mc'][0]['marketDefinition']['runners'][0]['status']
    market_timestamp = datetime.timestamp(market_datetime) * 1000
    return data_list, runner_id, runner_id_2, runner_name, runner_name_2, r1_result, market_timestamp


def get_list(runner_id, data_list, market_timestamp):
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
def odds_avg(runner_1, runner_2, r1_result):
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
    df_avg_odds = df.resample('2000ms').last()
    last_index = df_avg_odds.last_valid_index() + pd.Timedelta(2, 'sec')
    final_index = last_index + pd.Timedelta(6, 'min')
    df_datetime = pd.date_range(last_index, final_index, freq='2000ms')
    if r1_result == 'WINNER':
        df_ones = pd.DataFrame({'runner 1': np.ones(181), '1 - runner 2': np.ones(181), 'avg': np.ones(181)}, index=df_datetime)
        df_avg_odds = pd.concat([df_avg_odds, df_ones])
    else:
        df_zeros = pd.DataFrame({'runner 1': np.zeros(181), '1 - runner 2': np.zeros(181), 'avg': np.zeros(181)}, index=df_datetime)
        df_avg_odds = pd.concat([df_avg_odds, df_zeros])
    return df_avg_odds

def get_best_pricevol(runner_list, price, vol):
# Function to get best lay and back information
    back_list = []
    lay_list = []

    back_vol = []
    lay_vol = []

    for item in runner_list:
        if 'atl' in item[0]:
            if len(item[0]['atl']) > 1:
                temp = []
                for i in item[0]['atl']:
                    if i[1] != 0:
                        temp.append(i[0])
                if temp:
                    temp = np.array(temp)
                    back_list.append([temp.min(), item[1]])
                    back_vol.append([item[0]['atl'][temp.argmax()][1], item[1]])
            else:
                if item[0]['atl'][0][1] != 0:
                    back_list.append([item[0]['atl'][0][0], item[1]])
                    back_vol.append([item[0]['atl'][0][1], item[1]])

    for item in runner_list:
        if 'atb' in item[0]:
            if len(item[0]['atb']) > 1:
                temp = []
                for i in item[0]['atb']:
                    if i[1] != 0:
                        temp.append(i[0])
                if temp:
                    temp = np.array(temp)
                    lay_list.append([temp.max(), item[1]])
                    lay_vol.append([item[0]['atb'][temp.argmax()][1], item[1]])
            else:
                if item[0]['atb'][0][1] != 0:
                    lay_list.append([item[0]['atb'][0][0], item[1]])
                    lay_vol.append([item[0]['atb'][0][1], item[1]])

    del back_list[-1], lay_list[-1], back_vol[-1], lay_vol[-1]

    back_vol_arr = np.array(back_vol)
    lay_vol_arr = np.array(lay_vol)
    back_arr = np.array(back_list)
    lay_arr = np.array(lay_list)

    back_series = pd.Series(back_arr[:, 0], index=pd.to_datetime(back_arr[:, 1], unit='ms'))
    lay_series = pd.Series(lay_arr[:, 0], index=pd.to_datetime(lay_arr[:, 1], unit='ms'))
    back_vol_series = pd.Series(back_vol_arr[:, 0], index=pd.to_datetime(back_vol_arr[:, 1], unit='ms'))
    lay_vol_series = pd.Series(lay_vol_arr[:, 0], index=pd.to_datetime(lay_vol_arr[:, 1], unit='ms'))
# Apply hampel filter to remove outliers
    back_outliers = hampel(back_series, window_size=80)
    lay_outliers = hampel(lay_series, window_size=80)
    back_vol_outliers = hampel(back_vol_series, window_size=15)
    lay_vol_outliers = hampel(lay_vol_series, window_size=15)

    back_arr = np.delete(back_arr, back_outliers, axis=0)
    lay_arr = np.delete(lay_arr, lay_outliers, axis=0)
    back_vol_arr = np.delete(back_vol_arr, back_vol_outliers, axis=0)
    lay_vol_arr = np.delete(lay_vol_arr, lay_vol_outliers, axis=0)

    if price and not vol:
        return back_arr, lay_arr
    elif vol and not price:
        return back_vol_arr, lay_vol_arr

    return back_arr, lay_arr, back_vol_arr, lay_vol_arr

def best_available_df(runner_list, start, end):
# Get back and lay information and calculate spread and price up probability
    end_time = end + pd.Timedelta(120, 'sec')
    df_datetime = pd.date_range(start, end_time, freq='100ms')
    df_datetime = df_datetime.floor('100ms')
    dt_shape = np.zeros(df_datetime.shape)

    back_arr, lay_arr, back_vol_arr, lay_vol_arr = get_best_pricevol(runner_list, True, True)

    df = pd.DataFrame({'back': dt_shape, 'lay': dt_shape, 'back_vol': dt_shape, 'lay_vol': dt_shape}, index=df_datetime)

    for index, time in enumerate(back_arr[:, 1]):
        df['back'].loc[pd.to_datetime(round(time, -2), unit='ms')] = back_arr[index, 0]
    for index, time in enumerate(lay_arr[:, 1]):
        df['lay'].loc[pd.to_datetime(round(time, -2), unit='ms')] = lay_arr[index, 0]

    for index, time in enumerate(back_vol_arr[:, 1]):
        df['back_vol'].loc[pd.to_datetime(round(time, -2), unit='ms')] = back_vol_arr[index, 0]
    for index, time in enumerate(lay_vol_arr[:, 1]):
        df['lay_vol'].loc[pd.to_datetime(round(time, -2), unit='ms')] = lay_vol_arr[index, 0]

    df.replace(0, np.nan, inplace=True)
    df.interpolate(method='time', limit_direction='both', inplace=True)

    _2000ms = df.index.floor('2000ms')
    idx_back = df.groupby(_2000ms)['back'].idxmin()
    idx_lay = df.groupby(_2000ms)['lay'].idxmax()
    df = df.resample('2000ms').mean().assign(back=df.loc[idx_back]['back'].values,
                                            back_vol=df.loc[idx_back]['back_vol'].values,
                                            lay=df.loc[idx_lay]['lay'].values,
                                            lay_vol=df.loc[idx_lay]['lay_vol'].values)

    df_best = df.rolling('60S').mean()
    df_best = df_best.loc[start:end]
    # last_index = end + pd.Timedelta(1, 'sec')
    # final_index = last_index + pd.Timedelta(59, 'sec')
    # df_datetime_new = pd.date_range(last_index, final_index, freq='2000ms')
    # print(end, last_index)
    # if r1_result == 'WINNER':
    #     df_ones = pd.DataFrame({'back': np.repeat(1000, 60), 'back_vol': np.zeros(60), 'lay': np.repeat(1000, 60), 'lay_vol': np.repeat(0.001, 60)}, index=df_datetime_new)
    #     df_best = pd.concat([df_best, df_ones])
    # else:
    #     df_zeros = pd.DataFrame({'back': np.ones(60), 'back_vol': np.zeros(60), 'lay': np.ones(60), 'lay_vol': np.repeat(0.001, 60)}, index=df_datetime_new)
    #     df_best = pd.concat([df, df_zeros])
    df_best['back-lay avg'] = df_best[['back', 'lay']].mean(axis=1)
    df_best['spread'] = df_best['back'] - df_best['lay']
    df_best['vol diff'] = df_best['back_vol'] - df_best['lay_vol']
    df_best['uncertainty'] = df_best['spread'] / df_best['back-lay avg']
    df_pup = df_best['back_vol'] / ( df_best['back_vol'] + df_best['lay_vol'] )
    df_pup.name = 'pup'
    df_best = pd.concat([df_best, df_pup], axis=1)
    df_best.fillna(method='ffill', inplace=True)

    return df_best