# Based largely on code from Phil Tabor's tutorial - https://youtu.be/hlv79rcHws0?si=WouU8XbXytVISCQe
# some changes were made in order to utilise a continuous action space

import gymnasium as gym
import numpy as np
import os
import torch
import torch.nn as nn

from utils import pathJoin
from Memory import Memory
from torch.nn import (
    Linear,
    Softmax,
    LSTM,
    Sequential,
    init,
    Tanh,
    Parameter,
    functional as F,
)
from torch.distributions import Dirichlet
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SAVE_SUFFIX = "_ppo"
_OPTIMISER_SAVE_SUFFIX = "_optimiser_ppo"


def layerInit(layer, std=np.sqrt(2), biasConst=0.0):
    init.orthogonal_(layer.weight, std)
    init.constant_(layer.bias, biasConst)
    return layer


class CriticNetwork(nn.Module):
    """
    Creates a critic network class for PPO

    """

    def __init__(
        self,
        fc1_n: int,
        fc2_n: int,
        lstmHiddenSize: int,
        state_n: int,
        actions_n: int,
        modelName: str,
    ):
        super(CriticNetwork, self).__init__()
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.lstmHiddenSize = lstmHiddenSize
        self.actions_n = actions_n
        self.state_n = state_n
        self.modelName = modelName

        self.save_file_name = self.modelName + _SAVE_SUFFIX
        self.optimiser_save_file_name = self.modelName + _OPTIMISER_SAVE_SUFFIX
        self.lstm = LSTM(
            input_size=self.state_n,
            hidden_size=lstmHiddenSize,
            num_layers=1,
            batch_first=True,
        )
        self.fc1 = layerInit(Linear(self.lstmHiddenSize, self.fc1_n))
        self.fc2 = layerInit(Linear(self.fc1_n, self.fc2_n))
        self.fc3 = layerInit(Linear(self.fc2_n, 1))
        self.tanh = nn.Tanh()

    def forward(self, state: torch.Tensor):
        lstmOut, (hidden, _) = self.lstm(state.unsqueeze(1))
        valuation = self.tanh(self.fc1(hidden[-1]))
        valuation = self.tanh(self.fc2(valuation))
        valuation = valuation.squeeze(-1)
        valuation = self.fc3(valuation)

        return valuation


class ActorNetwork(nn.Module):
    """
    Creates an actor network class for PPO

    """

    def __init__(
        self,
        fc1_n: int,
        fc2_n: int,
        lstmHiddenSize: int,
        state_n: int,
        actions_n: int,
        modelName: str,
    ):
        super(ActorNetwork, self).__init__()
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.lstmHiddenSize = lstmHiddenSize
        self.state_n = state_n
        self.actions_n = actions_n
        self.modelName = modelName
        self.save_file_name = self.modelName + _SAVE_SUFFIX
        self.optimiser_save_file_name = self.modelName + _OPTIMISER_SAVE_SUFFIX

        # Map state to action
        self.lstm = LSTM(
            input_size=self.state_n,
            hidden_size=self.lstmHiddenSize,
            num_layers=1,
            batch_first=True,
        )

        self.fc1 = layerInit(Linear(self.lstmHiddenSize, self.fc1_n))
        self.fc2 = layerInit(Linear(self.fc1_n, self.fc2_n))
        self.alphaLayer = layerInit(Linear(self.fc2_n, actions_n), std=0.05)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        state = state.unsqueeze(1)  # adds sequence length dimension
        lstmOut, (hidden, _) = self.lstm(state)
        x = self.relu(self.fc1(hidden[-1]))
        x = self.relu(self.fc2(x))
        alpha = F.softplus(self.alphaLayer(x)) + 1e-3  # gotta be positive
        dist = torch.distributions.Dirichlet(alpha)
        return dist


class PPOAgent:
    """
    Creates an agent class for PPO

    """

    def __init__(
        self,
        alpha: float,  # actor learnign rate
        policyClip: float,
        gamma: float,  # discount factor
        actor_noise: float,  # noise for actor
        lstmHiddenSize: int,
        state_n: int,
        actions_n: int,
        batch_size: int,
        fc1_n: int,  # number of neurons in first hidden layer
        fc2_n: int,  # number of neurons in second hidden layer
        gaeLambda: int = 0.95,
        epochs=10,
        riskAversion=0,
        featureExtractor=None,
    ):
        self.alpha = alpha
        self.policyClip = policyClip
        self.gamma = gamma
        self.noise = actor_noise
        self.state_n = state_n
        self.lstmHiddenSize = lstmHiddenSize
        self.actions_n = actions_n
        self.batch_size = batch_size
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.gaeLambda = gaeLambda
        self.epochs = epochs
        self.memory = Memory(batchSize=batch_size)
        self.learn_step_count = 0
        self.time_step = 0
        self.riskAversion = riskAversion
        self.featureExtractor = featureExtractor

        self.actor = ActorNetwork(
            self.fc1_n,
            self.fc2_n,
            self.lstmHiddenSize,
            self.state_n,
            self.actions_n,
            modelName="actor",
        ).to(device)

        self.critic = CriticNetwork(
            self.fc1_n,
            self.fc2_n,
            self.lstmHiddenSize,
            self.state_n,
            self.actions_n,
            modelName="critic",
        ).to(device)

        self.optimizer = torch.optim.Adam(
            (
                list(self.actor.parameters())
                + list(self.critic.parameters())
                + list(featureExtractor.parameters())
            ),
            lr=self.alpha,
        )

    def select_action(self, observation=None, sampling=True):
        with torch.no_grad():
            state = torch.FloatTensor(observation).to(device)
            distribution = self.actor(state)
            if sampling:
                action = distribution.sample()
            else:
                action = distribution.mean
            criticValuation = self.critic(state)
            probabilities = distribution.log_prob(action)  # assumes independence
            action = torch.squeeze(action)
        # self.time_step += 1 # not actually necessary to care about luckily
        return (
            action,
            probabilities,
            criticValuation,
        )

    def store(self, state, action, probabilities, valuations, reward, done):
        self.memory.store(state, action, probabilities, valuations, reward, done)

    def train(self):  # joint training
        for _ in range(self.epochs):
            (
                stateArr,
                actionArr,
                oldProbArr,
                valsArr,
                rewardArr,
                donesArr,
                batches,
            ) = self.memory.generateBatches()

            values = valsArr
            advantage = np.zeros(len(rewardArr), dtype=np.float32)

            for t in range(len(rewardArr) - 1):
                discount = 1
                aT = 0
                for k in range(t, len(rewardArr) - 1):
                    aT += discount * (
                        rewardArr[k]
                        + self.gamma * values[k + 1] * (1 - int(donesArr[k]))
                        - values[k]
                    )
                    discount *= self.gamma * self.gaeLambda
                advantage[t] = aT
            advantage = torch.tensor(advantage).to(device)

            values = torch.tensor(values).to(device)
            for batch in batches:
                states = self.featureExtractor.forward(
                    torch.tensor(stateArr[batch], dtype=torch.float32)
                ).to(device)
                oldProbs = torch.tensor(oldProbArr[batch]).to(device)
                actions = torch.tensor(actionArr[batch]).to(device)

                dist = self.actor(states)
                criticValue = self.critic(states)

                criticValue = torch.squeeze(criticValue)

                newProbs = dist.log_prob(actions)
                oldProbs = oldProbs.squeeze()
                probRatio = torch.exp(newProbs - oldProbs)
                weightedProbs = advantage[batch] * probRatio
                weightedClippedProbs = (
                    torch.clamp(probRatio, 1 - self.policyClip, 1 + self.policyClip)
                    * advantage[batch]
                )
                actorLoss = -torch.min(weightedProbs, weightedClippedProbs).mean()

                returns = advantage[batch] + values[batch]
                criticLoss = (returns - criticValue) ** 2
                criticLoss = criticLoss.mean()

                totalLoss = actorLoss + 0.5 * criticLoss
                self.optimizer.zero_grad()
                totalLoss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]["params"], max_norm=0.5
                )
                self.optimizer.step()
        self.memory.clear()

    def save(self, save_dir: str):
        # print("### SAVING MODELS ###")

        torch.save(
            self.critic.state_dict(), pathJoin(save_dir, self.critic.save_file_name)
        )
        torch.save(
            self.optimizer.state_dict(),
            pathJoin(save_dir, self.critic.optimiser_save_file_name),
        )

        torch.save(
            self.actor.state_dict(), pathJoin(save_dir, self.actor.save_file_name)
        )

    def load(self, save_dir: str):
        self.critic.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.critic.save_file_name),
                weights_only=True,
            )
        )
        self.optimizer.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.critic.optimiser_save_file_name),
                weights_only=True,
            )
        )

        self.actor.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.actor.save_file_name),
                weights_only=True,
            )
        )
