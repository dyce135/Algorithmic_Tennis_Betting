import pandas as pd
import os
from os.path import join
import json
import numpy as np

runner_names = []

for i, file in enumerate(os.listdir('Wimbledon')):
    with open(join('Wimbledon/', file), 'r') as f:
        top = f.readline()
        top = json.loads(top)

    runner_name = top['mc'][0]['marketDefinition']['runners'][0]['name']
    runner_name_2 = top['mc'][0]['marketDefinition']['runners'][1]['name']

    runner_names.append(runner_name)
    runner_names.append(runner_name_2)

idx_players = pd.DataFrame({'player_name': runner_names})
idx_players.drop_duplicates(inplace=True)
players = idx_players['player_name'].tolist()

df_players = pd.read_csv('atp_players.csv', index_col=0)
df_players['full_name'] = df_players['name_first'] + ' ' + df_players['name_last']

id_list = []
name_list = []

for name in players:
    element = df_players[df_players['full_name'].str.casefold() == name.casefold()]
    p_id = element.index.values[0]
    name_full = element['full_name'].values[0]
    id_list.append(p_id)
    name_list.append(name_full)

df_dataset_players = pd.DataFrame({'name': name_list})
df_dataset_players.index = id_list
df_dataset_players.to_csv('dataset_player_id.csv')

df_new_features = pd.read_csv('new_features.csv', index_col=0)

df_nf_dataset = df_new_features.loc[id_list]
df_nf_dataset = df_nf_dataset[df_nf_dataset['server_points'] == 0]
df_nf_dataset = df_nf_dataset[df_nf_dataset['receiver_points'] == 0]
df_nf_dataset = pd.concat([df_nf_dataset, df_dataset_players], axis=1)
df_nf_dataset.to_csv('features_for_training.csv')
