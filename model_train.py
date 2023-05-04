import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import mean_absolute_error as mae
import seq2seq_model
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

df = pd.read_csv('Data/4105819v2474763.csv', index_col=0)
data = df.to_numpy()

train, test = train_test_split(data, test_size=0.2, shuffle=False)

model, history = seq2seq_model.seq2seq_fit(data, n_length=30, n_steps=3, features_out_num=1, features_in=range(8), features_out=range(1))
model.summary()

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('train_hist.csv')
model.save("seq2seq_model", save_format='tf')
