from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, GRU, RepeatVector, Dense, TimeDistributed, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/opt/rocm-5.3.0/lib'


def truncate(x, feature_cols=range(5), target_cols=range(5), train_len=825, test_len=165):
    in_, out_ = [], []
    for i in range(len(x) - train_len - test_len + 1):
        in_.append(x[i:(i + train_len), feature_cols].tolist())
        out_.append(x[(i + train_len):(i + train_len + test_len), target_cols].tolist())

    return np.array(in_), np.array(out_)


def seq2seq(input_dim, output_dim, latent_dim, learning_rate=0.001):
    input_train = Input(shape=(input_dim[1], input_dim[2]))
    output_train = Input(shape=(output_dim[1], output_dim[2]))
    encoder_last_h1, encoder_last_h2 = GRU(
        latent_dim, activation='elu', dropout=0.2, recurrent_dropout=0.2,
        return_sequences=False, return_state=True)(input_train)
    encoder_last_h1 = BatchNormalization(momentum=0.2)(encoder_last_h1)
    decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
    decoder = GRU(latent_dim, activation='elu', dropout=0.2, recurrent_dropout=0.2, return_state=False,
                  return_sequences=True)(
        decoder, initial_state=[encoder_last_h1])
    out = TimeDistributed(Dense(output_train.shape[2]))(decoder)
    model = Model(inputs=input_train, outputs=out)
    opt = Adam(lr=learning_rate, clipnorm=1)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])

    return model
