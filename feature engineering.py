import pandas as pd
import itertools
import warnings
import os
from os.path import join
import json

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset
# data = pd.read_csv("atp_matches_2023.csv")
# data2 = pd.read_csv("atp_matches_2022.csv")
# data3 = pd.read_csv("atp_matches_2021.csv")
# data4 = pd.read_csv("atp_matches_2020.csv")
# data5 = pd.read_csv("atp_matches_2019.csv")
# data6 = pd.read_csv("atp_matches_2018.csv")
data7 = pd.read_csv("atp_matches_2017.csv")
data8 = pd.read_csv("atp_matches_2016.csv")
data9 = pd.read_csv("atp_matches_2015.csv")
data10 = pd.read_csv("atp_matches_2014.csv")
data11 = pd.read_csv("atp_matches_2013.csv")
data12 = pd.read_csv("atp_matches_2012.csv")
data13 = pd.read_csv("atp_matches_2011.csv")

# data = pd.concat([data, data2, data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13])
data = pd.concat([data7,data8,data9,data10,data11,data12,data13])
data.to_csv("combined_csv",index=False)


def average_rank_point_difference(player):
    rank_point_diff_as_winner = data[data['winner_id'] == player]['winner_rank_points'] - data[data['winner_id'] == player]['loser_rank_points']
    rank_point_diff_as_loser = data[data['loser_id'] == player]['loser_rank_points'] - data[data['loser_id'] == player]['winner_rank_points']
    total_rank_point_diff = rank_point_diff_as_winner.sum() + rank_point_diff_as_loser.sum()
    total_matches = data[(data['winner_id'] == player) | (data['loser_id'] == player)].shape[0]
    return total_rank_point_diff / total_matches
# 2. Recent form
def recent_form(player, num_matches=10):
    recent_matches = data[((data['winner_id'] == player) | (data['loser_id'] == player))].tail(num_matches)
    wins = recent_matches[recent_matches['winner_id'] == player].shape[0]
    return wins / num_matches

# 3. Average aces per match
def average_aces(player):
    aces = data[(data['winner_id'] == player)]['w_ace'].sum() + data[(data['loser_id'] == player)]['l_ace'].sum()
    total_matches = data[(data['winner_id'] == player) | (data['loser_id'] == player)].shape[0]
    return aces / total_matches

# 4. Average double faults per match
def average_double_faults(player):
    dfs = data[(data['winner_id'] == player)]['w_df'].sum() + data[(data['loser_id'] == player)]['l_df'].sum()
    total_matches = data[(data['winner_id'] == player) | (data['loser_id'] == player)].shape[0]
    return dfs / total_matches

# 5. First serve win percentage
def first_serve_win_percentage(player):
    first_serves_won = data[(data['winner_id'] == player)]['w_1stWon'].sum() + data[(data['loser_id'] == player)]['l_1stWon'].sum()
    first_serves_made = data[(data['winner_id'] == player)]['w_1stIn'].sum() + data[(data['loser_id'] == player)]['l_1stIn'].sum()
    return first_serves_won / first_serves_made

# 6. Second serve win percentage
def second_serve_win_percentage(player):
    second_serves_won = data[(data['winner_id'] == player)]['w_2ndWon'].sum() + data[(data['loser_id'] == player)]['l_2ndWon'].sum()
    total_serves = data[(data['winner_id'] == player)]['w_svpt'].sum() + data[(data['loser_id'] == player)]['l_svpt'].sum()
    first_serves_made = data[(data['winner_id'] == player)]['w_1stIn'].sum() + data[(data['loser_id'] == player)]['l_1stIn'].sum()
    second_serves_made = total_serves - first_serves_made
    return second_serves_won / second_serves_made

# 7. Break point save percentage
def break_point_save_percentage(player):
    break_points_saved = data[(data['winner_id'] == player)]['w_bpSaved'].sum() + data[(data['loser_id'] == player)]['l_bpSaved'].sum()
    break_points_faced = data[(data['winner_id'] == player)]['w_bpFaced'].sum() + data[(data['loser_id'] == player)]['l_bpFaced'].sum()
    return break_points_saved / break_points_faced


# Calculate features for all players
unique_players = data['winner_id'].append(data['loser_id']).unique()
features_df = pd.DataFrame(columns=['player_id', 'recent_form','average_rank_point_difference', 'average_aces', 'average_double_faults', 'first_serve_win_percentage', 'second_serve_win_percentage', 'break_point_save_percentage'])

for player in unique_players:
    player_features = {
        'player_id': player,
        'recent_form': recent_form(player),
        'average_rank_point_difference': average_rank_point_difference(player),
        'average_aces': average_aces(player),
        'average_double_faults': average_double_faults(player),
        'first_serve_win_percentage': first_serve_win_percentage(player),
        'second_serve_win_percentage': second_serve_win_percentage(player),
        'break_point_save_percentage': break_point_save_percentage(player),
    }
    features_df = features_df.append(player_features, ignore_index=True)
# Generate all possible combinations of server_points and receiver_points
all_scores = list(itertools.product(range(5), repeat=2))

# Remove invalid score combinations (e.g., 3-3, 4-4)
valid_scores = [score for score in all_scores if score not in [(3, 3), (4, 4)]]

# Generate a new dataset containing all combinations of players and valid scores
combined_dataset = pd.DataFrame(columns=['player_id', 'server_points', 'receiver_points'])

for player_id in unique_players:
    for server_points, receiver_points in valid_scores:
        row = {
            'player_id': player_id,
            'server_points': server_points,
            'receiver_points': receiver_points
        }
        combined_dataset = combined_dataset.append(row, ignore_index=True)

features_df = features_df.dropna()

# Merge the datasets
merged_dataset = pd.merge(combined_dataset, features_df, on='player_id')

# Save the new features to a CSV file
merged_dataset.to_csv("new_features.csv", index=False)

print('saved')



