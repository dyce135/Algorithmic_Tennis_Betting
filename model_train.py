import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import mean_absolute_error as mae
import seq2seq_model
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

df = pd.read_csv('Data/Train/2249345v4423910.csv', index_col=0)
data = df.to_numpy()

train, test = train_test_split(data, test_size=0.4, shuffle=False)

# model, history = seq2seq_model.seq2seq_fit(data, n_length=56, n_steps=3, features_out_num=1, features_in=range(1), features_out=range(1), batch_size=16, epochs=10, fc_dim=20)
model, history = seq2seq_model.lstm_fit(data, features_in=range(14), features_out_num=14, lstm_dim_1=200, lstm_dim_2=300,
                                        features_out=range(14), n_steps=75, epochs=70, lr=0.0001, batch_size=16)
model.summary()

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('train_hist.csv')

model.save("lstm_model", save_format='tf')
