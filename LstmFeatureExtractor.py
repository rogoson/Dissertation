import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from utils import pathJoin
from torch.nn import Linear, LSTM, ReLU
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

_SAVE_SUFFIX = "_lstm"
_OPTIMISER_SAVE_SUFFIX = "_optimiser_lstm"


class LstmFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        numAssets,
        timeWindow,
        numFeatures,
        memorySize,
        lstmHiddenSize=128,
        lstmOutputSize=50,
    ):
        super(LstmFeatureExtractor, self).__init__(
            observation_space, features_dim=lstmOutputSize
        )
        self.lstmHiddenSize = lstmHiddenSize
        self.lstmOutputSize = lstmOutputSize
        self.numAssets = numAssets
        self.timeWindow = timeWindow
        self.numFeatures = numFeatures
        self.memorySize = memorySize

        self.lstm = LSTM(
            input_size=numFeatures,
            hidden_size=lstmHiddenSize,
            num_layers=1,
            batch_first=True,
        )
        self.fc = Linear(lstmHiddenSize, lstmOutputSize)
        self.relu = ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the output from the last time step.
        last_out = lstm_out[:, -1, :]
        features = self.fc(last_out)
        features = self.relu(features)
        return features
