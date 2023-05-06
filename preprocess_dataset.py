import preprocessing as P
import score_inference as score
import os
from os.path import join
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


for i, file in enumerate(os.listdir('Wimbledon')):
    current_file = join('Wimbledon', file)
    print('Processing ' + current_file + '...')
    data_list, runner_id, runner_id_2, r1_name, r2_name, r1_result, market_timestamp = P.get_data(current_file)

    runner_list_1 = P.get_list(runner_id, data_list, market_timestamp)
    runner_list_2 = P.get_list(runner_id_2, data_list, market_timestamp)
    runner_odds_1 = P.convert_odds(runner_list_1)
    runner_odds_2 = P.convert_odds(runner_list_2)
    df = P.odds_avg(runner_odds_1, runner_odds_2, r1_result)

    start = df.first_valid_index()
    end = df.last_valid_index()

    df_score = score.get_score_time_series(df['avg'])
    df_enhanced = score.get_enhanced_markov_probs(df_score, r1_name, r2_name)
    df_markov = pd.concat([df_score, df_enhanced], axis=1)

    index_extend_markov = pd.date_range(df_markov.last_valid_index() + pd.Timedelta('2min'),
                                        df_markov.last_valid_index() + pd.Timedelta('10min'), freq='2min')
    df_zeros = pd.DataFrame(
        {'r1_setscore': np.repeat(420, 5), 'r2_setscore': np.repeat(420, 5), 'markov_odds': np.repeat(420, 5),
         'match_score': np.repeat(420, 5), 'enhanced_markov_odds': np.repeat(420, 5)}, index=index_extend_markov)
    df_markov_data = pd.concat([df_markov, df_zeros], axis=0)
    df_markov_data.replace(420, method='ffill', inplace=True)
    df_markov_data = df_markov_data.resample('2000ms').last()
    df_markov_data.replace(np.nan, method='ffill', inplace=True)
    df_markov_data = df_markov_data[start:end]

    df_runner_1 = P.best_available_df(runner_list_1, start, end)
    df_runner_2 = P.best_available_df(runner_list_2, start, end)

    df_avg = df['avg']
    df_blodds = (1 / df_runner_1['back-lay avg'] + 1 - 1 / df_runner_2['back-lay avg']) / 2
    df_odds = pd.concat([df_avg, df_blodds], axis=1)
    df_odds.columns = ['ltp odds', 'back lay odds']

    df_r1_stw = pd.get_dummies(3 - df_markov_data['r1_setscore'], dtype=float)
    if 0 not in df_r1_stw.columns:
        df_zeros_score = pd.DataFrame({'0': np.zeros(df_r1_stw.shape[0])}, index=df_r1_stw.index)
        df_r1_stw = df_zeros_score.join(df_r1_stw)
    df_r1_stw.columns = ['r1_0', 'r1_1', 'r1_2', 'r1_3']
    df_r2_stw = pd.get_dummies(3 - df_markov_data['r2_setscore'], dtype=float)
    if 0 not in df_r2_stw.columns:
        df_zeros_score = pd.DataFrame({'0': np.zeros(df_r2_stw.shape[0])}, index=df_r2_stw.index)
        df_r2_stw = df_zeros_score.join(df_r2_stw)
    print(df_r2_stw.head())
    df_r2_stw.columns = ['r2_0', 'r2_1', 'r2_2', 'r2_3']
    df_total = pd.DataFrame({'lpt odds': df_odds['ltp odds'], 'r1 spread': df_runner_1['uncertainty'],
                              'r1 pup': df_runner_1['pup'], 'r2 spread': df_runner_2['uncertainty'], 
                              'r2 pup': df_runner_2['pup'], 'enhanced_markov': df_markov_data['enhanced_markov_odds']}, index=df_odds.index)
    df_total = df_total.join(df_r1_stw)
    df_total = df_total.join(df_r2_stw)

    df_total.replace(to_replace=np.nan, method='ffill', inplace=True)
    df_total.to_csv(join('Data/', str(runner_id) + 'v' + str(runner_id_2) + '.csv'))
    print('Saved to ' + join('Data/', str(runner_id) + 'v' + str(runner_id_2) + '.csv'))

quit()
