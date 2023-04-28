import pandas as pd
import numpy as np
import markov_sim as markov

def get_serve_prob(first_odds):
    df = pd.read_csv('markov_serve_prob.csv', index_col=0)
    difference = np.absolute(df.index.to_numpy() - first_odds)
    index = difference.argmin()
    r1 = df.iloc[index]
    r2 = 1.29 - r1
    return r1, r2


def get_score_time_series(p, q, df_odds, server):
    setscore = '0-0'
    gamescore = '0-0'

    df_odds = df_odds.resample('10000ms').last()
    odds_arr = df_odds.to_numpy()
    set_trans = markov.set_nextscore()
    match_trans = markov.match_nextscore()
    mis, sis, gis, tbis = initiate_markov_states()

    # r1_gamescore = np.zeros(odds_arr.shape, dtype=int)
    r1_setscore = np.empty(odds_arr.shape, dtype=int)
    # r2_gamescore = np.empty(odds_arr.shape, dtype=int)
    r2_setscore = np.empty(odds_arr.shape, dtype=int)

    for i, odds in enumerate(odds_arr):
        # game_temp = gamescore
        setscore, gamescore, p, q = calc_score(p, q, setscore, gamescore, odds, set_trans, match_trans, mis,
                                               sis, gis, tbis)
        # r1_gamescore[i] = gamescore.split('-')[0]
        # r2_gamescore[i] = gamescore.split('-')[1]
        r1_setscore[i] = setscore.split('-')[0]
        r2_setscore[i] = setscore.split('-')[1]
        if r1_setscore[i] == 3 or r2_setscore[i] == 3:
            print('Match finished at ' + setscore)
            break
        # if gamescore != game_temp and gamescore != '6-6':
        #     if server == 1:
        #         server = 0
        #     else:
        #         server = 1
        #     p, q = recal_pointprobs(odds, gamescore, setscore, server, p, q, mis, sis, gis, tbis)

    df = pd.DataFrame(
        {'r1_setscore': r1_setscore, 'r2_setscore': r2_setscore},
        index=df_odds.index)
    df.iloc[i + 1:] = 0
    df.replace(to_replace=0, method='ffill', inplace=True)
    df = df.resample('2000ms').last().shift(int(- 360 / 2))
    df.replace(to_replace=np.nan, method='ffill', inplace=True)

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


def calc_score(p, q, setscore, gamescore, implied_odds, set_transitions, match_transitions, mis, sis, gis,
               tbis):
    # nextsetv1 = set_nextscore(set_transitions, gamescore, 1)
    # nextsetv2 = set_nextscore(set_transitions, gamescore, -1)
    nextmatchv1 = match_nextscore(match_transitions, setscore, 1)
    nextmatchv2 = match_nextscore(match_transitions, setscore, -1)

    # if nextsetv1 == 'SETv1' and nextsetv2 == 'SETv2':
    #     nextsetv1 = '0-0'
    #     nextsetv2 = '0-0'
    # elif nextsetv1 == 'SETv1':
    #     nextsetv1 = '0-0'
    #     nextmatchv2 = setscore
    # elif nextsetv2 == 'SETv2':
    #     nextsetv2 = '0-0'
    #     nextmatchv1 = setscore
    # else:
    #     nextmatchv1 = setscore
    #     nextmatchv2 = setscore

    mis1 = mis.copy()
    sis1 = sis.copy()
    gis1 = gis.copy()
    tbis1 = tbis.copy()
    mis2 = mis.copy()
    sis2 = sis.copy()
    gis2 = gis.copy()
    tbis2 = tbis.copy()

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

    m_avg = (m1 + m2) / 2
    gap = (m1 - m2) * 0.4
    m1 = m_avg + gap
    m2 = m_avg - gap

    if implied_odds >= m1:
        # gamescore = nextsetv1
        setscore = nextmatchv1
    elif implied_odds <= m2:
        # gamescore = nextsetv2
        setscore = nextmatchv2

    return setscore, gamescore, p, q

