import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/opt/rocm-5.3.0/lib'


# LSTM model class
class LstmModel(k.Model):

    # Constructor with layers
    def __init__(self, size, **kwargs):
        self.kernel_dev = kwargs.get('kernel_stddev', 0.01)
        self.kernel = k.initializers.RandomNormal(mean=0, stddev=self.kernel_dev)
        self.bias = kwargs.get('kernel_initializer', k.initializers.Zeros())
        self.drop = kwargs.get('drop', 0.5)
        self.nodes = kwargs.get('nodes', 1024)
        self.normalise = kwargs.get('normalise', True)
        super().__init__()

    # Call function to connect layers
    def call(self, inputs):
        return out