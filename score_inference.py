import pandas as pd
import numpy as np
import enhanced
import markov_sim as markov
import copy


def get_serve_prob(first_odds):
    df = pd.read_csv('markov_serve_prob.csv', index_col=0)
    difference = np.absolute(df.index.to_numpy() - first_odds)
    index = difference.argmin()
    r1 = df.iloc[index]
    r2 = 1.29 - r1
    return r1, r2


def get_score_time_series(df_odds):
    setscore = '0-0'
    gamescore = '0-0'
    k = 0
    m = np.nan
    gap1 = 0
    gap2 = 0
    set_changing = False

    df_odds = df_odds.resample('120000ms').last()
    odds_arr = df_odds.to_numpy()
    match_trans = markov.match_nextscore()
    mis, sis, gis, tbis = initiate_markov_states()
    r1, r2 = get_serve_prob(odds_arr[0])

    r1_setscore = np.empty(odds_arr.shape, dtype=int)
    r2_setscore = np.empty(odds_arr.shape, dtype=int)
    markov_odds = np.empty(odds_arr.shape)

    stationary = ['3-0', '3-2', '3-1', '1-3', '2-3', '0-3']
    r1_stationary = ['3-0', '3-2', '3-1']

    for i, odds in enumerate(odds_arr):
        set_temp = copy.copy(setscore)
        m_temp = copy.copy(m)
        gap1_temp = copy.copy(gap1)
        gap2_temp = copy.copy(gap2)
        setscore, m, gap1, gap2 = calc_score(r1, r2, setscore, gamescore, odds,
                                             match_trans, mis, sis, gis, tbis)
        # If change in score detected
        if setscore in stationary:
            if set_changing is True:
                r1_setscore[j:i] = set_temp.split('-')[0]
                r2_setscore[j:i] = set_temp.split('-')[1]
                markov_odds[j:i] = m
            r1_setscore[i] = setscore.split('-')[0]
            r2_setscore[i] = setscore.split('-')[1]
            if setscore in r1_stationary:
                markov_odds[i] = 1
            else:
                markov_odds[i] = 0
            print('Match finished at ' + setscore)
            break
        elif setscore != set_temp and set_changing is False:
            set_changing = True
            set_no_change = copy.copy(set_temp)
            gap1_no_change = copy.copy(gap1_temp)
            gap2_no_change = copy.copy(gap2_temp)
            m_no_change = copy.copy(m_temp)
            k = k - 6
            j = copy.copy(i)
        # Wait 10 time steps
        elif set_changing is True and (i - k) > 0:
            k = k + 2
            r1_setscore[i] = set_no_change.split('-')[0]
            r2_setscore[i] = set_no_change.split('-')[1]
            markov_odds[i] = m_no_change
        # If change in score and waited 10 timesteps
        elif set_changing is True and k == i:
            # If odds still within thresholds
            if odds > m_no_change + gap1_no_change or odds < m_no_change - gap2_no_change:
                r1_setscore[j:i + 1] = setscore.split('-')[0]
                r2_setscore[j:i + 1] = setscore.split('-')[1]
                markov_odds[j:i + 1] = m
            else:
                r1_setscore[j:i + 1] = set_no_change.split('-')[0]
                r2_setscore[j:i + 1] = set_no_change.split('-')[1]
                markov_odds[j:i + 1] = m_no_change
                setscore = copy.copy(set_no_change)
                m = copy.copy(m_no_change)
            set_changing = False
        elif set_changing is False:
            r1_setscore[i] = setscore.split('-')[0]
            r2_setscore[i] = setscore.split('-')[1]
            markov_odds[i] = m
        k = k + 1

    df = pd.DataFrame(
        {'r1_setscore': r1_setscore, 'r2_setscore': r2_setscore, 'markov_odds': markov_odds},
        index=df_odds.index)
    df.iloc[i + 1:] = 999
    df.replace(to_replace=999, method='ffill', inplace=True)
    df['match_score'] = df['r1_setscore'].astype(str) + '-' + df['r2_setscore'].astype(str)
    # df = df.resample('2000ms').last()
    # df.replace(to_replace=np.nan, method='bfill', inplace=True)
    # df.replace(to_replace=np.nan, method='ffill', inplace=True)

    return df


def initiate_markov_states():
    matrix = np.zeros((1, 17))
    game_initial_state = pd.DataFrame(data=matrix, columns=markov.col_row_names)
    game_initial_state.at[0, "0-0"] = 1
    matrix = np.zeros((1, 54))
    tb_initial_state = pd.DataFrame(data=matrix, columns=markov.col_row_names2)
    tb_initial_state.at[0, "0-0"] = 1
    matrix = np.zeros((1, 41))
    set_initial_sate = pd.DataFrame(data=matrix, columns=markov.col_row_names3)
    set_initial_sate.at[0, "0-0"] = 1
    matrix = np.zeros((1, 17))
    match_initial_state = pd.DataFrame(data=matrix, columns=markov.col_row_names4)
    match_initial_state.at[0, "0-0"] = 1

    return match_initial_state, set_initial_sate, game_initial_state, tb_initial_state


# def recal_pointprobs(odds, gamescore, setscore, server, p, q, mis, sis, gis, tbis):
#     odds_temp = 0
#     error_old = 1
#
#     if server == 1:
#         serve_temp = p
#         while abs(odds - odds_temp) <= error_old:
#             mis1 = mis.copy()
#             sis1 = sis.copy()
#             gis1 = gis.copy()
#             tbis1 = tbis.copy()
#             error_old = abs(odds - odds_temp)
#             if odds_temp < odds:
#                 serve_temp = serve_temp + 0.001
#             else:
#                 serve_temp = serve_temp - 0.001
#             odds_temp = \
#             markov.tennis_model(serve_temp, q, setscore, gamescore, mis1, sis1, gis1, tbis1)['r1_win'].values[0]
#         p = serve_temp
#     else:
#         serve_temp = q
#         while abs(odds - odds_temp) <= error_old:
#             mis1 = mis.copy()
#             sis1 = sis.copy()
#             gis1 = gis.copy()
#             tbis1 = tbis.copy()
#             error_old = abs(odds - odds_temp)
#             if odds_temp < odds:
#                 serve_temp = serve_temp - 0.001
#             else:
#                 serve_temp = serve_temp + 0.001
#             odds_temp = markov.tennis_model(p, serve_temp, setscore, gamescore, mis1, sis1, gis1, tbis1)[
#                 'r1_win'].values
#         q = serve_temp
#
#     return p, q


def set_nextscore(set_transitions, score, winner):
    next_score = set_transitions.columns[(set_transitions == winner).loc[score]]
    return next_score[0]


def match_nextscore(match_transitions, score, winner):
    next_score = match_transitions.columns[(match_transitions == winner).loc[score]]
    return next_score[0]


def calc_score(p, q, setscore, gamescore, implied_odds, match_transitions, mis, sis, gis,
               tbis):

    nextmatchv1 = match_nextscore(match_transitions, setscore, 1)
    nextmatchv2 = match_nextscore(match_transitions, setscore, -1)

    mis1 = mis.copy()
    sis1 = sis.copy()
    gis1 = gis.copy()
    tbis1 = tbis.copy()
    mis2 = mis.copy()
    sis2 = sis.copy()
    gis2 = gis.copy()
    tbis2 = tbis.copy()
    mis3 = mis.copy()
    sis3 = sis.copy()
    gis3 = gis.copy()
    tbis3 = tbis.copy()

    if nextmatchv1 == ('3-2') and nextmatchv2 == ('2-3'):
        m1 = 1
        m2 = 0
    elif nextmatchv1 == '3-2' or nextmatchv1 == '3-1' or nextmatchv1 == '3-0':
        m1 = 1
        m2 = markov.tennis_model(p, q, nextmatchv2, gamescore, mis2, sis2, gis2, tbis2)['r1_win'].values[0]
    elif nextmatchv2 == '2-3' or nextmatchv2 == '1-3' or nextmatchv2 == '0-3':
        m1 = markov.tennis_model(p, q, nextmatchv1, gamescore, mis1, sis1, gis1, tbis1)['r1_win'].values[0]
        m2 = 0
    else:
        m1 = markov.tennis_model(p, q, nextmatchv1, gamescore, mis1, sis1, gis1, tbis1)['r1_win'].values[0]
        m2 = markov.tennis_model(p, q, nextmatchv2, gamescore, mis2, sis2, gis2, tbis2)['r1_win'].values[0]

    m = markov.tennis_model(p, q, setscore, gamescore, mis3, sis3, gis3, tbis3)['r1_win'].values[0]
    gap1 = m1 - m
    gap2 = m - m2

    if implied_odds >= m1:
        setscore = nextmatchv1
    elif implied_odds <= m2:
        setscore = nextmatchv2

    # print(setscore, m1, m2, implied_odds, m)

    return setscore, m, gap1, gap2

def get_enhanced_markov_probs(df_scores, r1_name, r2_name):

    features_data = pd.read_csv('features_for_training.csv')

    r1_features = features_data[features_data['name'].str.casefold() == r1_name.casefold()]
    r2_features = features_data[features_data['name'].str.casefold() == r2_name.casefold()]

    mis, sis, gis, tbis = markov.initiate_markov_states()
    score_arr = df_scores['match_score'].to_numpy()

    player1_features = {
        'server_points': 0,
        'receiver_points': 0,
        'recent_form': r1_features['recent_form'].values[0],
        'average_rank_point_difference': r1_features['average_rank_point_difference'].values[0],
        'average_aces': r1_features['average_aces'].values[0],
        'average_double_faults': r1_features['average_double_faults'].values[0],
        'break_point_save_percentage': r1_features['break_point_save_percentage'].values[0]
    }

    player2_features = {
        'server_points': 0,
        'receiver_points': 0,
        'recent_form': r2_features['recent_form'].values[0],
        'average_rank_point_difference': r2_features['average_rank_point_difference'].values[0],
        'average_aces': r2_features['average_aces'].values[0],
        'average_double_faults': r2_features['average_double_faults'].values[0],
        'break_point_save_percentage': r2_features['break_point_save_percentage'].values[0]
    }

    markov_odds = np.empty(score_arr.shape)
    stationary = ['3-0', '3-2', '3-1', '1-3', '2-3', '0-3']
    r1_winner_list = ['3-0', '3-2', '3-1']
    for i, score in enumerate(score_arr[:-1]):
        mis1 = mis.copy()
        sis1 = sis.copy()
        gis1 = gis.copy()
        tbis1 = tbis.copy()
        if score in stationary:
            if score in r1_winner_list:
                markov_odds[i:] = 1
            else:
                markov_odds[-1] = 0
            break
        m = enhanced.tennis_model(player1_features, player2_features, score, '0-0', mis1, sis1, gis1, tbis1)['r1_win'].values[0]
        markov_odds[i] = m

    df_markov = pd.DataFrame({'enhanced_markov_odds': markov_odds}, index=df_scores.index)

    return df_markov
