import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import mean_absolute_error as mae
import seq2seq_model
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

df = pd.read_csv('Data/Train/4105819v2474763.csv', index_col=0)
data = df.to_numpy()

train, test = train_test_split(data, test_size=0.4, shuffle=False)

# model, history = seq2seq_model.seq2seq_fit(data, n_length=60, n_steps=3, features_out_num=1, features_in=range(8), features_out=range(1), epochs=10, fc_dim=20)
model, history = seq2seq_model.lstm_fit(data, features_in=range(8), features_out_num=8, features_out=range(8), n_steps=60, epochs=30)
model.summary()

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('train_hist.csv')

model.save("seq2seq_model_single", save_format='tf')
