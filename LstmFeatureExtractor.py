import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from utils import pathJoin
from torch.nn import Linear, LSTM, Tanh
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

_SAVE_SUFFIX = "_lstm"
_OPTIMISER_SAVE_SUFFIX = "_optimiser_lstm"


class LstmFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        timeWindow,
        numFeatures,
        lstmHiddenSize=128,
        lstmOutputSize=50,
    ):
        super(LstmFeatureExtractor, self).__init__(
            observation_space=None, features_dim=lstmOutputSize
        )
        self.lstmHiddenSize = lstmHiddenSize
        self.lstmOutputSize = lstmOutputSize
        self.timeWindow = timeWindow

        self.lstm = LSTM(
            input_size=numFeatures,
            hidden_size=lstmHiddenSize,
            num_layers=1,
            batch_first=True,
        )
        self.fc1 = Linear(lstmHiddenSize, lstmHiddenSize)
        self.fc2 = Linear(lstmHiddenSize, lstmHiddenSize)
        self.fc3 = Linear(lstmHiddenSize, lstmOutputSize)
        self.tanh = Tanh()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        features = self.tanh(self.fc1(hidden[-1]))
        features = self.tanh(self.fc2(features))
        features = self.tanh(self.fc3(features))
        return features
