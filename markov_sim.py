import numpy as np
import pandas as pd

# Constants
global col_row_names, col_row_names2, col_row_names3, col_row_names4
col_row_names = ["0-0", "0-15", "15-0", "15-15", "30-0", "0-30", "40-0", "30-15", "15-30", "0-40", "40-15", "15-40",
                 "30-30(DEUCE)", "40-30(A-40)", "30-40(40-A)", "Win", "Lose"]
col_row_names2 = ["0-0", "0-1", "1-0", "1-1",
                  "2-0", "0-2", "3-0", "2-1",
                  "1-2", "0-3", "4-0", "3-1",
                  "2-2", "1-3", "0-4", "5-0",
                  "4-1", "3-2", "2-3", "1-4",
                  "0-5", "5-1", "4-2", "3-3",
                  "2-4", "1-5", "5-2", "4-3", "3-4",
                  "2-5", "5-3", "4-4", "3-5", "5-4",
                  "4-5", "5-5", "6-5", "5-6",
                  "6-6", "SETv1", "SETv2", "6-0",
                  "6-1", "6-2", "6-3", "6-4", "4-6",
                  "3-6", "2-6", "1-6", "0-6", "7-7", "7-6", "6-7"]
col_row_names3 = ["0-0", "0-1", "1-0", "1-1",
                  "2-0", "0-2", "3-0", "2-1",
                  "1-2", "0-3", "4-0", "3-1",
                  "2-2", "1-3", "0-4", "5-0",
                  "4-1", "3-2", "2-3", "1-4",
                  "0-5", "5-1", "4-2", "3-3",
                  "2-4", "1-5", "5-2", "4-3", "3-4",
                  "2-5", "5-3", "4-4", "3-5", "5-4",
                  "4-5", "5-5", "6-5", "5-6",
                  "6-6", "SETv1", "SETv2"]
col_row_names4 = ["0-0", "0-1", "1-0", "1-1", "2-0", "0-2", "2-1", "1-2", "2-2", "3-0", "0-3", "3-1", "1-3", "3-2", "2-3", "V1", "V2"]
matrix = np.zeros((1, 17))
game_initial_state = pd.DataFrame(data=matrix, columns=col_row_names)
game_initial_state.at[0, "0-0"] = int(1)
matrix = np.zeros((1, 54))
tb_initial_state = pd.DataFrame(data=matrix, columns=col_row_names2)
tb_initial_state.at[0, "0-0"] = 1
matrix = np.zeros((1, 41))
set_initial_sate = pd.DataFrame(data=matrix, columns=col_row_names3)
set_initial_sate.at[0, "0-0"] = 1
matrix = np.zeros((1, 17))
match_initial_state = pd.DataFrame(data=matrix, columns=col_row_names4)  #
match_initial_state.at[0, "0-0"] = 1


def game_trans_matrix(ppoint_server):
    ppoint_ret = 1 - ppoint_server
    matrix = np.zeros((17, 17))
    tMat1 = pd.DataFrame(data=matrix, index=col_row_names, columns=col_row_names)
    tMat1.at["0-0", "15-0"] = ppoint_server
    tMat1.at["15-0", "30-0"] = ppoint_server
    tMat1.at["0-15", "15-15"] = ppoint_server
    tMat1.at["30-0", "40-0"] = ppoint_server
    tMat1.at["15-15", "30-15"] = ppoint_server
    tMat1.at["0-30", "15-30"] = ppoint_server
    tMat1.at["40-0", "Win"] = ppoint_server
    tMat1.at["30-15", "40-15"] = ppoint_server
    tMat1.at["40-15", "Win"] = ppoint_server
    tMat1.at["40-30(A-40)", "Win"] = ppoint_server
    tMat1.at["0-40", "15-40"] = ppoint_server
    tMat1.at["15-40", "30-40(40-A)"] = ppoint_server
    tMat1.at["30-40(40-A)", "30-30(DEUCE)"] = ppoint_server
    tMat1.at["15-30", "30-30(DEUCE)"] = ppoint_server
    tMat1.at["30-30(DEUCE)", "40-30(A-40)"] = ppoint_server

    tMat1.at["0-0", "0-15"] = ppoint_ret
    tMat1.at["15-0", "15-15"] = ppoint_ret
    tMat1.at["0-15", "0-30"] = ppoint_ret
    tMat1.at["30-0", "30-15"] = ppoint_ret
    tMat1.at["15-15", "15-30"] = ppoint_ret
    tMat1.at["0-30", "0-40"] = ppoint_ret
    tMat1.at["40-0", "40-15"] = ppoint_ret
    tMat1.at["30-15", "30-30(DEUCE)"] = ppoint_ret
    tMat1.at["40-15", "40-30(A-40)"] = ppoint_ret
    tMat1.at["40-30(A-40)", "30-30(DEUCE)"] = ppoint_ret
    tMat1.at["0-40", "Lose"] = ppoint_ret
    tMat1.at["15-40", "Lose"] = ppoint_ret
    tMat1.at["30-40(40-A)", "Lose"] = ppoint_ret
    tMat1.at["15-30", "15-40"] = ppoint_ret
    tMat1.at["30-30(DEUCE)", "30-40(40-A)"] = ppoint_ret

    tMat1.at["Lose", "Lose"] = 1
    tMat1.at["Win", "Win"] = 1

    return tMat1


def prob_game(ppoint_server, s_game):
    matrix = game_trans_matrix(ppoint_server)
    temp = matrix
    for i in range(1000):
        matrix = np.dot(matrix, matrix)
    matrix = pd.DataFrame(data=matrix, index=col_row_names, columns=col_row_names)
    probs = np.dot(s_game, matrix)
    return probs


def tie_break(ppoint_srv1, ppoint_srv2):
    states = ["0-0", "0-1", "1-0", "1-1", "2-0", "0-2", "3-0", "2-1",
              "1-2", "0-3", "4-0", "3-1",
              "2-2", "1-3", "0-4", "5-0",
              "4-1", "3-2", "2-3", "1-4",
              "0-5", "5-1", "4-2", "3-3",
              "2-4", "1-5", "5-2", "4-3", "3-4",
              "2-5", "5-3", "4-4", "3-5", "5-4",
              "4-5", "5-5", "6-5", "5-6",
              "6-6", "SETv1", "SETv2", "6-0",
              "6-1", "6-2", "6-3", "6-4", "4-6",
              "3-6", "2-6", "1-6", "0-6", "7-7", "7-6", "6-7"]
    matrix = np.zeros((54, 54))
    tMat2 = pd.DataFrame(data=matrix, index=states, columns=states)
    tMat2.at["0-0", "1-0"] = ppoint_srv1
    tMat2.at["3-0", "4-0"] = ppoint_srv1
    tMat2.at["2-1", "3-1"] = ppoint_srv1
    tMat2.at["1-2", "2-2"] = ppoint_srv1
    tMat2.at["0-3", "1-3"] = ppoint_srv1
    tMat2.at["4-0", "5-0"] = ppoint_srv1
    tMat2.at["3-1", "4-1"] = ppoint_srv1
    tMat2.at["2-2", "3-2"] = ppoint_srv1
    tMat2.at["1-3", "2-3"] = ppoint_srv1
    tMat2.at["0-4", "1-4"] = ppoint_srv1
    tMat2.at["6-1", "SETv1"] = ppoint_srv1
    tMat2.at["5-2", "6-2"] = ppoint_srv1
    tMat2.at["4-3", "5-3"] = ppoint_srv1
    tMat2.at["3-4", "4-4"] = ppoint_srv1
    tMat2.at["2-5", "3-5"] = ppoint_srv1
    tMat2.at["1-6", "2-6"] = ppoint_srv1
    tMat2.at["6-2", "SETv1"] = ppoint_srv1
    tMat2.at["5-3", "6-3"] = ppoint_srv1
    tMat2.at["4-4", "5-4"] = ppoint_srv1
    tMat2.at["3-5", "4-5"] = ppoint_srv1
    tMat2.at["2-6", "3-6"] = ppoint_srv1
    tMat2.at["6-5", "SETv1"] = ppoint_srv1
    tMat2.at["5-6", "6-6"] = ppoint_srv1
    tMat2.at["6-6", "7-6"] = ppoint_srv1

    tMat2.at["0-0", "0-1"] = 1 - ppoint_srv1
    tMat2.at["3-0", "3-1"] = 1 - ppoint_srv1
    tMat2.at["2-1", "2-2"] = 1 - ppoint_srv1
    tMat2.at["1-2", "1-3"] = 1 - ppoint_srv1
    tMat2.at["0-3", "0-4"] = 1 - ppoint_srv1
    tMat2.at["4-0", "4-1"] = 1 - ppoint_srv1
    tMat2.at["3-1", "3-2"] = 1 - ppoint_srv1
    tMat2.at["2-2", "2-3"] = 1 - ppoint_srv1
    tMat2.at["1-3", "1-4"] = 1 - ppoint_srv1
    tMat2.at["0-4", "0-5"] = 1 - ppoint_srv1
    tMat2.at["6-1", "6-2"] = 1 - ppoint_srv1
    tMat2.at["5-2", "5-3"] = 1 - ppoint_srv1
    tMat2.at["4-3", "4-4"] = 1 - ppoint_srv1
    tMat2.at["3-4", "3-5"] = 1 - ppoint_srv1
    tMat2.at["2-5", "2-6"] = 1 - ppoint_srv1
    tMat2.at["1-6", "SETv2"] = 1 - ppoint_srv1
    tMat2.at["6-2", "6-3"] = 1 - ppoint_srv1
    tMat2.at["5-3", "5-4"] = 1 - ppoint_srv1
    tMat2.at["4-4", "4-5"] = 1 - ppoint_srv1
    tMat2.at["3-5", "3-6"] = 1 - ppoint_srv1
    tMat2.at["2-6", "SETv2"] = 1 - ppoint_srv1
    tMat2.at["6-5", "6-6"] = 1 - ppoint_srv1
    tMat2.at["5-6", "SETv2"] = 1 - ppoint_srv1
    tMat2.at["3-4", "3-5"] = 1 - ppoint_srv1
    tMat2.at["6-6", "6-7"] = 1 - ppoint_srv1

    tMat2.at["1-0", "1-1"] = ppoint_srv2
    tMat2.at["0-1", "0-2"] = ppoint_srv2
    tMat2.at["2-0", "2-1"] = ppoint_srv2
    tMat2.at["1-1", "1-2"] = ppoint_srv2
    tMat2.at["0-2", "0-3"] = ppoint_srv2
    tMat2.at["0-3", "0-4"] = ppoint_srv2
    tMat2.at["5-0", "5-1"] = ppoint_srv2
    tMat2.at["4-1", "4-2"] = ppoint_srv2
    tMat2.at["3-2", "3-3"] = ppoint_srv2
    tMat2.at["2-3", "2-4"] = ppoint_srv2
    tMat2.at["1-4", "1-5"] = ppoint_srv2
    tMat2.at["0-5", "0-6"] = ppoint_srv2
    tMat2.at["6-0", "6-1"] = ppoint_srv2
    tMat2.at["5-1", "5-2"] = ppoint_srv2
    tMat2.at["4-2", "5-2"] = ppoint_srv2
    tMat2.at["3-3", "3-4"] = ppoint_srv2
    tMat2.at["2-4", "2-5"] = ppoint_srv2
    tMat2.at["1-5", "1-6"] = ppoint_srv2
    tMat2.at["0-6", "SETv2"] = ppoint_srv2
    tMat2.at["6-3", "6-4"] = ppoint_srv2
    tMat2.at["5-4", "5-5"] = ppoint_srv2
    tMat2.at["4-5", "4-6"] = ppoint_srv2
    tMat2.at["3-6", "SETv2"] = ppoint_srv2
    tMat2.at["6-4", "6-5"] = ppoint_srv2
    tMat2.at["5-5", "5-6"] = ppoint_srv2
    tMat2.at["4-6", "SETv2"] = ppoint_srv2
    tMat2.at["6-7", "SETv2"] = ppoint_srv2
    tMat2.at["7-6", "7-7"] = ppoint_srv2
    tMat2.at["4-2", "4-3"] = ppoint_srv2
    tMat2.at["7-7", "5-6"] = ppoint_srv2

    tMat2.at["1-0", "2-0"] = 1 - ppoint_srv2
    tMat2.at["0-1", "1-1"] = 1 - ppoint_srv2
    tMat2.at["2-0", "3-0"] = 1 - ppoint_srv2
    tMat2.at["1-1", "2-1"] = 1 - ppoint_srv2
    tMat2.at["0-2", "1-2"] = 1 - ppoint_srv2
    tMat2.at["0-3", "1-3"] = 1 - ppoint_srv2
    tMat2.at["5-0", "6-0"] = 1 - ppoint_srv2
    tMat2.at["4-1", "5-1"] = 1 - ppoint_srv2
    tMat2.at["3-2", "4-2"] = 1 - ppoint_srv2
    tMat2.at["2-3", "3-3"] = 1 - ppoint_srv2
    tMat2.at["1-4", "2-4"] = 1 - ppoint_srv2
    tMat2.at["0-5", "1-5"] = 1 - ppoint_srv2
    tMat2.at["6-0", "SETv1"] = 1 - ppoint_srv2
    tMat2.at["5-1", "6-1"] = 1 - ppoint_srv2
    tMat2.at["4-2", "5-2"] = 1 - ppoint_srv2
    tMat2.at["3-3", "4-3"] = 1 - ppoint_srv2
    tMat2.at["2-4", "3-4"] = 1 - ppoint_srv2
    tMat2.at["1-5", "2-5"] = 1 - ppoint_srv2
    tMat2.at["0-6", "1-6"] = 1 - ppoint_srv2
    tMat2.at["6-3", "SETv1"] = 1 - ppoint_srv2
    tMat2.at["5-4", "6-4"] = 1 - ppoint_srv2
    tMat2.at["4-5", "5-5"] = 1 - ppoint_srv2
    tMat2.at["3-6", "4-6"] = 1 - ppoint_srv2
    tMat2.at["6-4", "SETv1"] = 1 - ppoint_srv2
    tMat2.at["5-5", "6-5"] = 1 - ppoint_srv2
    tMat2.at["4-6", "5-6"] = 1 - ppoint_srv2
    tMat2.at["3-6", "4-6"] = 1 - ppoint_srv2
    tMat2.at["6-7", "7-7"] = 1 - ppoint_srv2
    tMat2.at["7-6", "SETv1"] = 1 - ppoint_srv2
    tMat2.at["7-7", "6-5"] = 1 - ppoint_srv2

    # Set stationary states
    tMat2.at["SETv1", "SETv1"] = 1
    tMat2.at["SETv2", "SETv2"] = 1
    return tMat2


def prob_tie(ppoint_srv1, ppoint_srv2, s_tb):
    matrix = tie_break(ppoint_srv1, ppoint_srv2)
    for i in range(10000):
        matrix = np.dot(matrix, matrix)
    matrix = pd.DataFrame(data=matrix, index=col_row_names2, columns=col_row_names2)
    probs = np.dot(s_tb, matrix)
    return probs


def set(pwin1, pwin2, ptie1):
    matrix = np.zeros((41, 41))
    tMat3 = pd.DataFrame(data=matrix, index=col_row_names3, columns=col_row_names3)
    tMat3.at["0-0", "1-0"] = pwin1
    tMat3.at["2-0", "3-0"] = pwin1
    tMat3.at["1-1", "2-1"] = pwin1
    tMat3.at["0-2", "1-2"] = pwin1
    tMat3.at["4-0", "5-0"] = pwin1
    tMat3.at["3-1", "4-1"] = pwin1
    tMat3.at["2-2", "3-2"] = pwin1
    tMat3.at["1-3", "2-3"] = pwin1
    tMat3.at["0-4", "1-4"] = pwin1
    tMat3.at["5-1", "SETv1"] = pwin1
    tMat3.at["4-2", "5-2"] = pwin1
    tMat3.at["3-3", "4-3"] = pwin1
    tMat3.at["2-4", "3-4"] = pwin1
    tMat3.at["1-5", "2-5"] = pwin1
    tMat3.at["5-3", "SETv1"] = pwin1
    tMat3.at["4-4", "5-4"] = pwin1
    tMat3.at["3-5", "4-5"] = pwin1
    tMat3.at["5-5", "6-5"] = pwin1

    tMat3.at["0-0", "0-1"] = 1 - pwin1
    tMat3.at["2-0", "2-1"] = 1 - pwin1
    tMat3.at["1-1", "1-2"] = 1 - pwin1
    tMat3.at["0-2", "0-3"] = 1 - pwin1
    tMat3.at["4-0", "4-1"] = 1 - pwin1
    tMat3.at["3-1", "3-2"] = 1 - pwin1
    tMat3.at["2-2", "2-3"] = 1 - pwin1
    tMat3.at["1-3", "1-4"] = 1 - pwin1
    tMat3.at["0-4", "0-5"] = 1 - pwin1
    tMat3.at["5-1", "5-2"] = 1 - pwin1
    tMat3.at["4-2", "4-3"] = 1 - pwin1
    tMat3.at["3-3", "3-4"] = 1 - pwin1
    tMat3.at["2-4", "2-5"] = 1 - pwin1
    tMat3.at["1-5", "SETv2"] = 1 - pwin1
    tMat3.at["5-3", "5-4"] = 1 - pwin1
    tMat3.at["4-4", "4-5"] = 1 - pwin1
    tMat3.at["3-5", "SETv2"] = 1 - pwin1
    tMat3.at["5-5", "5-6"] = 1 - pwin1

    tMat3.at["1-0", "1-1"] = pwin2
    tMat3.at["0-1", "0-2"] = pwin2
    tMat3.at["3-0", "3-1"] = pwin2
    tMat3.at["2-1", "2-2"] = pwin2
    tMat3.at["1-2", "1-3"] = pwin2
    tMat3.at["0-3", "0-4"] = pwin2
    tMat3.at["5-0", "5-1"] = pwin2
    tMat3.at["4-1", "4-2"] = pwin2
    tMat3.at["3-2", "3-3"] = pwin2
    tMat3.at["2-3", "2-4"] = pwin2
    tMat3.at["1-4", "1-5"] = pwin2
    tMat3.at["0-5", "SETv2"] = pwin2
    tMat3.at["5-2", "5-3"] = pwin2
    tMat3.at["4-3", "4-4"] = pwin2
    tMat3.at["3-4", "3-5"] = pwin2
    tMat3.at["2-5", "SETv2"] = pwin2
    tMat3.at["5-4", "5-5"] = pwin2
    tMat3.at["4-5", "SETv2"] = pwin2
    tMat3.at["5-6", "SETv2"] = pwin2
    tMat3.at["6-5", "6-6"] = pwin2

    tMat3.at["1-0", "2-0"] = 1 - pwin2
    tMat3.at["0-1", "1-1"] = 1 - pwin2
    tMat3.at["3-0", "4-0"] = 1 - pwin2
    tMat3.at["2-1", "3-1"] = 1 - pwin2
    tMat3.at["1-2", "2-2"] = 1 - pwin2
    tMat3.at["0-3", "1-3"] = 1 - pwin2
    tMat3.at["5-0", "SETv1"] = 1 - pwin2
    tMat3.at["4-1", "5-1"] = 1 - pwin2
    tMat3.at["3-2", "4-2"] = 1 - pwin2
    tMat3.at["2-3", "3-3"] = 1 - pwin2
    tMat3.at["1-4", "2-4"] = 1 - pwin2
    tMat3.at["0-5", "1-5"] = 1 - pwin2
    tMat3.at["5-2", "SETv1"] = 1 - pwin2
    tMat3.at["4-3", "5-3"] = 1 - pwin2
    tMat3.at["3-4", "4-4"] = 1 - pwin2
    tMat3.at["2-5", "3-5"] = 1 - pwin2
    tMat3.at["5-4", "SETv1"] = 1 - pwin2
    tMat3.at["4-5", "5-5"] = 1 - pwin2
    tMat3.at["5-6", "6-6"] = 1 - pwin2
    tMat3.at["6-5", "SETv1"] = 1 - pwin2

    # Set stationary states
    tMat3.at["SETv1", "SETv1"] = 1
    tMat3.at["SETv2", "SETv2"] = 1

    # Set tie-break cases
    tMat3.at["6-6", "SETv1"] = ptie1
    tMat3.at["6-6", "SETv2"] = 1 - ptie1
    return tMat3


def prob_set(pwin1, pwin2, ptie1, s_set):
    matrix = set(pwin1, pwin2, ptie1)
    for i in range(1000):
        matrix = np.dot(matrix, matrix)
    matrix = pd.DataFrame(data=matrix, index=col_row_names3, columns=col_row_names3)
    probs = np.dot(s_set, matrix)
    return probs


def match(pset_v1):
    pset_v2 = 1 - pset_v1
    matrix = np.zeros((17, 17))
    tMat4 = pd.DataFrame(data=matrix, index=col_row_names4, columns=col_row_names4)
    tMat4.at["0-0", "1-0"] = pset_v1
    tMat4.at["1-0", "2-0"] = pset_v1
    tMat4.at["0-1", "1-1"] = pset_v1
    tMat4.at["1-1", "2-1"] = pset_v1
    tMat4.at["2-2", "3-2"] = pset_v1
    tMat4.at["1-2", "2-2"] = pset_v1
    tMat4.at["2-0", "3-0"] = pset_v1
    tMat4.at["0-2", "1-2"] = pset_v1
    tMat4.at["2-1", "3-1"] = pset_v1

    tMat4.at["0-0", "0-1"] = pset_v2
    tMat4.at["1-0", "1-1"] = pset_v2
    tMat4.at["0-1", "0-2"] = pset_v2
    tMat4.at["1-1", "1-2"] = pset_v2
    tMat4.at["2-2", "2-3"] = pset_v2
    tMat4.at["1-2", "1-3"] = pset_v2
    tMat4.at["2-0", "2-1"] = pset_v2
    tMat4.at["0-2", "0-3"] = pset_v2
    tMat4.at["2-1", "2-2"] = pset_v2

    # Set stationary states
    tMat4.at["3-0", "3-0"] = 1
    tMat4.at["3-1", "3-1"] = 1
    tMat4.at["3-2", "3-2"] = 1
    tMat4.at["0-3", "0-3"] = 1
    tMat4.at["1-3", "1-3"] = 1
    tMat4.at["2-3", "2-3"] = 1

    tMat4.at["V1", "V1"] = 1
    tMat4.at["V2", "V2"] = 1

    return tMat4


def prob_match(pset_v1, s_match):
    matrix = match(pset_v1)
    for i in range(1000):
        matrix = np.dot(matrix, matrix)
    matrix = pd.DataFrame(data=matrix, index=col_row_names4, columns=col_row_names4)
    print(matrix)
    probs = np.dot(s_match, matrix)
    return probs


def predict1(gamescore, pwin1, pwin2, ptie1, pset_v1, match_initial_state, set_initial_sate, game_initial_state,
             tb_initial_state):
    s1match = match_initial_state
    s1set = set_initial_sate
    s1game = game_initial_state
    s1tb = tb_initial_state
    s1set.at[0, "0-0"] = 0
    s1set.at[0, gamescore] = 1

    ans = prob_set(pwin1, pwin2, ptie1, s1set)
    ans = pd.DataFrame(data=ans, columns=col_row_names3)
    temp = ans.at[0, "SETv1"]
    temp1 = ans.at[0, "SETv2"]

    s1match.at[0, "1-0"] = temp
    s1match.at[0, "0-1"] = temp1
    s1match.at[0, "0-0"] = 0

    prob_entire_match = prob_match(pset_v1, s1match)

    return prob_entire_match


def predict2(setscore, gamescore, pwin1, pwin2, ptie1, pset_v1, match_initial_state, set_initial_sate,
             game_initial_state, tb_initial_state):
    s1match = match_initial_state
    s1set = set_initial_sate
    s1game = game_initial_state
    s1tb = tb_initial_state

    s1match.at[0, "0-0"] = 0
    s1match.at[0, setscore] = 1
    s1set.at[0, "0-0"] = 0
    s1set.at[0, gamescore] = 1

    ans = prob_set(pwin1, pwin2, ptie1, s1set)
    ans = pd.DataFrame(data=ans, columns=col_row_names3)
    temp = ans.at[0, "SETv1"]
    temp1 = ans.at[0, "SETv2"]

    if (setscore == "1-0"):
        s1match.at[0, "2-0"] = temp
        s1match.at[0, "1-1"] = temp1
        s1match.at[0, "1-0"] = 0

    if (setscore == "0-1"):
        s1match.at[0, "1-1"] = temp
        s1match.at[0, "0-2"] = temp1
        s1match.at[0, "0-1"] = 0

    prob_entire_match = prob_match(pset_v1, s1match)

    return prob_entire_match


def predict3(setscore, gamescore, pwin1, pwin2, ptie1, pset_v1, match_initial_state, set_initial_sate, game_initial_state,
             tb_initial_state):
    s1match = match_initial_state
    s1set = set_initial_sate
    s1game = game_initial_state
    s1tb = tb_initial_state

    s1match.at[0, "0-0"] = 0
    s1match.at[0, setscore] = 1
    s1set.at[0, "0-0"] = 0
    s1set.at[0, gamescore] = 1

    ans = prob_set(pwin1, pwin2, ptie1, s1set)
    ans = pd.DataFrame(data=ans, columns=col_row_names3)
    temp = ans.at[0, "SETv1"]
    temp1 = ans.at[0, "SETv2"]

    if (setscore == "2-0"):
        s1match.at[0, "3-0"] = temp
        s1match.at[0, "2-1"] = temp1
        s1match.at[0, "2-0"] = 0

    if (setscore == "0-2"):
        s1match.at[0, "1-2"] = temp
        s1match.at[0, "0-3"] = temp1
        s1match.at[0, "0-2"] = 0

    if (setscore == "1-1"):
        s1match.at[0, "2-1"] = temp
        s1match.at[0, "1-2"] = temp1
        s1match.at[0, "1-1"] = 0

    prob_entire_match = prob_match(pset_v1, s1match)

    return prob_entire_match


def predict4(setscore, gamescore, pwin1, pwin2, ptie1, pset_v1, match_initial_state, set_initial_sate, game_initial_state,
             tb_initial_state):
    s1match = match_initial_state
    s1set = set_initial_sate
    s1game = game_initial_state
    s1tb = tb_initial_state

    s1match.at[0, "0-0"] = 0
    s1match.at[0, setscore] = 1
    s1set.at[0, "0-0"] = 0
    s1set.at[0, gamescore] = 1

    ans = prob_set(pwin1, pwin2, ptie1, s1set)
    ans = pd.DataFrame(data=ans, columns=col_row_names3)
    temp = ans.at[0, "SETv1"]
    temp1 = ans.at[0, "SETv2"]

    if (setscore == "2-1"):
        s1match.at[0, "3-1"] = temp
        s1match.at[0, "2-2"] = temp1
        s1match.at[0, "2-1"] = 0

    if (setscore == "1-2"):
        s1match.at[0, "2-2"] = temp
        s1match.at[0, "1-3"] = temp1
        s1match.at[0, "1-2"] = 0

    prob_entire_match = prob_match(pset_v1, s1match)

    return prob_entire_match

def predict5(gamescore, pwin1, pwin2, ptie1, pset_v1, match_initial_state, set_initial_sate, game_initial_state,
             tb_initial_state):
    s1match = match_initial_state
    s1set = set_initial_sate
    s1game = game_initial_state
    s1tb = tb_initial_state

    setscore = "2-2"

    s1match.at[0, "0-0"] = 0
    s1match.at[0, setscore] = 1
    s1set.at[0, "0-0"] = 0
    s1set.at[0, gamescore] = 1

    ans = prob_set(pwin1, pwin2, ptie1, s1set)
    ans = pd.DataFrame(data=ans, columns=col_row_names3)
    temp = ans.at[0, "SETv1"]
    temp1 = ans.at[0, "SETv2"]

    s1match.at[0, "3-2"] = temp
    s1match.at[0, "2-3"] = temp1
    s1match.at[0, "2-2"] = 0

    prob_entire_match = prob_match(pset_v1, s1match)
    return prob_entire_match


def tennis_model(ppoint_srv1, ppoint_srv2, setscore, gamescore, match_initial_state, set_initial_sate,
                 game_initial_state, tb_initial_state):
    ans = prob_game(ppoint_srv1, game_initial_state)
    ans = pd.DataFrame(data=ans, columns=col_row_names)
    temp = ans.at[0, "Win"]
    ans = prob_game(ppoint_srv2, game_initial_state)
    ans = pd.DataFrame(data=ans, columns=col_row_names)
    temp1 = ans.at[0, "Win"]
    ans = prob_tie(ppoint_srv1, ppoint_srv2, tb_initial_state)
    ans = pd.DataFrame(data=ans, columns=col_row_names2)
    temp2 = ans.at[0, "SETv1"]
    ans = prob_set(temp, temp1, temp2, set_initial_sate)
    ans = pd.DataFrame(data=ans, columns=col_row_names3)
    temp3 = ans.at[0, "SETv1"]

    if setscore == "0-0":
        prob_entire_match = predict1(gamescore, temp, temp1, temp2, temp3, match_initial_state, set_initial_sate,
                                     game_initial_state, tb_initial_state)
    if setscore == "1-0" or setscore == "0-1":
        prob_entire_match = predict2(setscore, gamescore, temp, temp1, temp2, temp3, match_initial_state,
                                     set_initial_sate, game_initial_state, tb_initial_state)
    if setscore == "2-0" or setscore == "0-2" or setscore == "1-1":
        prob_entire_match = predict3(setscore, gamescore, temp, temp1, temp2, temp3, match_initial_state,
                                     set_initial_sate, game_initial_state, tb_initial_state)
    if setscore == "2-1" or setscore == "1-2":
        prob_entire_match = predict4(setscore, gamescore, temp, temp1, temp2, temp3, match_initial_state,
                                     set_initial_sate, game_initial_state, tb_initial_state)
    if setscore == "2-2":
        prob_entire_match = predict5(gamescore, temp, temp1, temp2, temp3, match_initial_state,
                                     set_initial_sate, game_initial_state, tb_initial_state)

    ans = pd.DataFrame(data=prob_entire_match, columns=col_row_names4)
    ans['r1_win'] = ans['3-0'] + ans['3-1'] + ans['3-2']
    ans['r2_win'] = ans['0-3'] + ans['1-3'] + ans['2-3']

    return ans

#
# ans = tennis_model(0.6, 0.6, "0-0", "0-0", match_initial_state, set_initial_sate, game_initial_state, tb_initial_state)
# print(ans)
