import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, GRU, RepeatVector, Dense, TimeDistributed
from tensorflow.keras import Sequential
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/opt/rocm-5.3.0/lib'


def seq2seq(input_dim, output_dim):
    model = Sequential()
    # Encoder
    model.add(LSTM(units=input_dim, return_sequences=False, activation = 'tanh'))
    model.add(Dense(150, activation="relu") )
    #Use "RepeatVector" to copy N copies of Encoder's output (last time step) as Decoder's N inputs
    model.add(RepeatVector(output_dim))
    # Decoder (second LSTM)
    model.add(LSTM(units=input_dim, activation = 'tanh', return_sequences=True) )
    # TimeDistributed is to ensure consistency between Dense and Decoder
    model.add(TimeDistributed(Dense(output_dim, activation="linear")) )

    return model


