import preprocessing as P
import score_inference as score
import os
from os.path import join
import pandas as pd

for i, file in enumerate(os.listdir('Wimbledon')):
    current_file = join('Wimbledon', file)
    print('Processing ' + current_file + '...')
    data_list, runner_id, runner_id_2, r1_result, market_timestamp = P.get_data(file)

    runner_list_1 = P.get_list(runner_id, data_list, market_timestamp)
    runner_list_2 = P.get_list(runner_id_2, data_list, market_timestamp)
    runner_odds_1 = P.convert_odds(runner_list_1)
    runner_odds_2 = P.convert_odds(runner_list_2)
    df = P.odds_avg(runner_odds_1, runner_odds_2, r1_result)

    first_odds = df['avg'].iloc[0:300].mean()
    r1, r2 = score.get_serve_prob(first_odds)
    r1 = r1.values[0]
    r2 = r2.values[0]
    df_score = score.get_score_time_series(r1, r2, df['avg'], server=1)

    start = df.first_valid_index()
    end = df.last_valid_index()
    df_runner_1 = P.best_available_df(runner_list_1, start, end)
    df_runner_2 = P.best_available_df(runner_list_2, start, end)

    df_avg = df['avg']
    df_blodds = (1 / df_runner_1['back-lay avg'] + 1 - 1 / df_runner_2['back-lay avg']) / 2
    df_odds = pd.concat([df_avg, df_blodds], axis=1)
    df_odds.columns = ['ltp odds', 'back lay odds']

    df_total = pd.DataFrame({'lpt odds': df_odds['ltp odds'], 'r1 spread': df_runner_1['uncertainty'],
                             'r1 pup': df_runner_1['pup'], 'r2 spread': df_runner_2['uncertainty'],
                             'r2 pup': df_runner_2['pup'], 'r1_setscore': 3 - df_score['r1_setscore'],
                             'r2_setscore': 3 - df_score['r2_setscore']}, index=df_odds.index)

    df_total.to_csv(join('Data/', str(runner_id) + 'v' + str(runner_id_2) + '.csv'))

    print('Saved to ' + join('Data/', str(runner_id) + 'v' + str(runner_id_2) + '.csv'))
