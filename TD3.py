import gymnasium as gym
import numpy as np
import os
import torch
import torch.nn as nn

from torch.nn import Linear, LSTM
from torch.nn.functional import relu, mse_loss
from torch.optim import Adam

from Memory import ReplayBufferTwo
from utils import pathJoin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SAVE_SUFFIX = "_td3"
_OPTIMISER_SAVE_SUFFIX = "_optimiser_td3"


class CriticNetwork(nn.Module):
    """
    Creates a critic network class for TD3

    """

    def __init__(
        self,
        fc1_n: int,
        fc2_n: int,
        lstmHiddenSize: int,
        state_n: int,
        actions_n: int,
        model_name: str,
    ):
        super(CriticNetwork, self).__init__()
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.state_n = state_n
        self.actions_n = actions_n
        self.model_name = model_name

        self.save_file_name = self.model_name + _SAVE_SUFFIX
        self.optimiser_save_file_name = self.model_name + _OPTIMISER_SAVE_SUFFIX

        # Take in concatenation of states and actions, output Q value
        input_n = self.state_n + self.actions_n
        self.lstm = LSTM(
            input_size=input_n,
            hidden_size=lstmHiddenSize,
            num_layers=1,
            batch_first=True,
        )
        self.fc1 = Linear(lstmHiddenSize, self.fc1_n)
        self.fc2 = Linear(self.fc1_n, self.fc2_n)
        self.fc3 = Linear(self.fc2_n, 1)
        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        # Concat to fit input shape (state_n + actions_n)
        state_action = torch.cat([state, action.to(device)], 1)
        lstmOut, (hidden, _) = self.lstm(state_action)
        q = self.relu(self.fc1(hidden[-1]))
        q = self.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class ActorNetwork(nn.Module):
    """
    Creates an actor network class for TD3

    """

    def __init__(
        self,
        fc1_n: int,
        fc2_n: int,
        lstmHiddenSize: int,
        state_n: int,
        actions_n: int,
        model_name: str,
    ):
        super(ActorNetwork, self).__init__()
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.lstmHiddenSize = lstmHiddenSize
        self.state_n = state_n
        self.actions_n = actions_n
        self.model_name = model_name
        self.save_file_name = self.model_name + _SAVE_SUFFIX
        self.optimiser_save_file_name = self.model_name + _OPTIMISER_SAVE_SUFFIX

        self.lstm = LSTM(
            input_size=self.state_n,
            hidden_size=self.lstmHiddenSize,
            num_layers=1,
            batch_first=True,
        )

        # Map state to action
        self.fc1 = Linear(self.lstmHiddenSize, self.fc1_n)
        self.fc2 = Linear(self.fc1_n, self.fc2_n)
        self.fc3 = Linear(self.fc2_n, actions_n)
        self.tanh = nn.Tanh()

    def forward(self, state):
        state = state.unsqueeze(1)  # adds sequence length dimension
        lstmOut, (hidden, _) = self.lstm(state)
        x = self.tanh(self.fc1(hidden[-1]))
        action = relu(self.fc2(action))
        action = torch.tanh(
            self.fc3(action)
        )  # Use tanh to bind to action space [-1, 1]

        return action


class td3Agent:
    """
    Creates an agent class for TD3

    """

    def __init__(
        self,
        env: gym.Env,
        alpha: float,  # actor learnign rate
        beta: float,  # critic learning rate
        gamma: float,  # discount factor
        tau: float,  # interpolation parameter
        actor_noise: float,  # noise for actor exploration
        target_noise: float,  # noise for target networks
        state_n: int,
        actions_n: int,
        lstmHiddenSize: int,
        batch_size: int,
        fc1_n: int,  # number of neurons in first hidden layer
        fc2_n: int,  # number of neurons in second hidden layer
        actor_update_freq: int = 2,  # Update actor every n steps
        warmup: int = 500,
        max_size: int = 10000,
        numberOfUpdates: int = 5,
        riskAversion=0,
    ):
        self.env = env
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.noise = actor_noise
        self.target_noise = target_noise
        self.state_n = state_n
        self.actions_n = actions_n
        self.batch_size = batch_size
        self.lstmHiddenSize = lstmHiddenSize
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.actor_update_freq = actor_update_freq
        self.warmup = warmup
        self.max_size = max_size
        self.numberOfUpdates = numberOfUpdates
        self.riskAversion = riskAversion

        self.memory = ReplayBufferTwo(max_size, state_n, actions_n)
        self.learn_step_count = 0
        self.time_step = 0

        self.actor = ActorNetwork(
            self.fc1_n,
            self.fc2_n,
            self.state_n,
            self.actions_n,
            self.lstmHiddenSize,
            model_name="actor",
        ).to(device)
        self.actor_optimiser = Adam(self.actor.parameters(), lr=self.alpha)

        self.critic_1 = CriticNetwork(
            self.fc1_n,
            self.fc2_n,
            self.state_n,
            self.actions_n,
            self.lstmHiddenSize,
            model_name="critic_1",
        ).to(device)
        self.critic_1_optimiser = Adam(self.critic_1.parameters(), lr=self.beta)

        self.critic_2 = CriticNetwork(
            self.fc1_n,
            self.fc2_n,
            self.state_n,
            self.actions_n,
            self.lstmHiddenSize,
            model_name="critic_2",
        ).to(device)
        self.critic_2_optimiser = Adam(self.critic_2.parameters(), lr=self.beta)

        self.target_actor = ActorNetwork(
            self.fc1_n,
            self.fc2_n,
            self.state_n,
            self.actions_n,
            self.lstmHiddenSize,
            model_name="target_actor",
        ).to(device)
        self.target_actor_optimiser = Adam(self.actor.parameters(), lr=self.alpha)

        self.target_critic_1 = CriticNetwork(
            self.fc1_n,
            self.fc2_n,
            self.state_n,
            self.actions_n,
            self.lstmHiddenSize,
            model_name="target_critic_1",
        ).to(device)
        self.target_critic_1_optimiser = Adam(self.critic_1.parameters(), lr=self.beta)

        self.target_critic_2 = CriticNetwork(
            self.fc1_n,
            self.fc2_n,
            self.state_n,
            self.actions_n,
            self.lstmHiddenSize,
            model_name="target_critic_2",
        ).to(device)
        self.target_critic_2_optimiser = Adam(self.critic_2.parameters(), lr=self.beta)

        self.update_network(self.critic_1, self.target_critic_1)
        self.update_network(self.critic_2, self.target_critic_2)
        self.update_network(self.actor, self.target_actor)

    def select_action(self, observation, runBestModel=False):
        # Compare WITH and WITHOUT warmup
        if (self.time_step < self.warmup) and not runBestModel:
            # Pick random action when warming up to populate replay buffer
            action = np.random.normal(scale=self.noise, size=(self.actions_n))
        else:
            state = torch.FloatTensor(observation).to(device)
            action = self.actor(state)
            action = torch.squeeze(action)  # Convert to NDArray
        action = action + np.random.normal(
            scale=self.noise
        )  # Add noise to action for exploration
        action = torch.from_numpy(action).clamp(-1, 1)  # Clamp to action space [-1, 1]
        self.time_step += 1

        return action.cpu().data.numpy().flatten()

    def update_network(self, network, target_network):
        """
        Perform soft update on networks
        """
        for param, target_param in zip(
            network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        new_state: torch.Tensor,
        done: bool,
    ):
        self.memory.store(state, action, reward, new_state, done)

    def learn(self):
        # If there aren't enough experiences stored to fairly sample
        # self.numberOfUpdates updates, use as many as possible
        amountOfExperienceStored = self.memory.pointer // self.batch_size
        for update in range(min(amountOfExperienceStored, self.numberOfUpdates)):
            states, actions, rewards, new_states, done_arr = (
                self.memory.sample(self.batch_size, False)
                if self.memory.memory_full
                else self.memory.sample(self.batch_size, True)
            )
            states = states.to(device)
            states = states.view(-1, states.shape[-1]).T
            actions = actions.to(device)
            rewards = rewards.to(device).unsqueeze(1)
            new_states = new_states.to(device)
            new_states = new_states.view(-1, new_states.shape[-1]).T
            done_arr = done_arr.to(device).unsqueeze(1)
            # torch.autograd.set_detect_anomaly(True)
            with torch.no_grad():
                target_actions = self.target_actor(new_states)
                target_actions += np.random.normal(scale=self.target_noise)
                target_actions = torch.FloatTensor(target_actions.cpu()).clamp(
                    -1, 1
                )  # Clamp to action space [-1, 1]

                q1 = self.target_critic_1(new_states, target_actions)
                q2 = self.target_critic_2(new_states, target_actions)

                target_q = torch.min(q1, q2)  # Take min to reduce overestimation bias
                target_q = rewards + ~done_arr * self.gamma * target_q

            q1 = self.critic_1(states, actions)
            q2 = self.critic_2(states, actions)

            critic_1_loss = mse_loss(target_q, q1)
            critic_2_loss = mse_loss(target_q, q2)
            critic_loss = critic_1_loss + critic_2_loss

            self.critic_1_optimiser.zero_grad()
            self.critic_2_optimiser.zero_grad()
            critic_loss.backward()
            self.critic_1_optimiser.step()
            self.critic_2_optimiser.step()

            self.learn_step_count += 1
            # update actor every other time step
            if self.learn_step_count % self.actor_update_freq == 0:
                new_actions = self.actor(states)
                critic_1_value = self.critic_1(states, new_actions)
                actor_loss = critic_1_value.mean() * -1
                self.actor_optimiser.zero_grad()
                actor_loss.backward()
                self.actor_optimiser.step()

                self.update_network(self.critic_1, self.target_critic_1)
                self.update_network(self.critic_2, self.target_critic_2)
                self.update_network(self.actor, self.target_actor)

    def save(self, save_dir: str):
        print("### SAVING MODELS ###")

        torch.save(
            self.critic_1.state_dict(), pathJoin(save_dir, self.critic_1.save_file_name)
        )
        torch.save(
            self.critic_1_optimiser.state_dict(),
            pathJoin(save_dir, self.critic_1.optimiser_save_file_name),
        )

        torch.save(
            self.target_critic_1.state_dict(),
            pathJoin(save_dir, self.target_critic_1.save_file_name),
        )
        torch.save(
            self.target_critic_1_optimiser.state_dict(),
            pathJoin(save_dir, self.target_critic_1.optimiser_save_file_name),
        )

        torch.save(
            self.critic_2.state_dict(), pathJoin(save_dir, self.critic_2.save_file_name)
        )
        torch.save(
            self.critic_2_optimiser.state_dict(),
            pathJoin(save_dir, self.critic_2.optimiser_save_file_name),
        )

        torch.save(
            self.target_critic_2.state_dict(),
            pathJoin(save_dir, self.target_critic_2.save_file_name),
        )
        torch.save(
            self.target_critic_2_optimiser.state_dict(),
            pathJoin(save_dir, self.target_critic_2.optimiser_save_file_name),
        )

        torch.save(
            self.actor.state_dict(), pathJoin(save_dir, self.actor.save_file_name)
        )
        torch.save(
            self.actor_optimiser.state_dict(),
            pathJoin(save_dir, self.actor.optimiser_save_file_name),
        )

        torch.save(
            self.target_actor.state_dict(),
            pathJoin(save_dir, self.target_actor.save_file_name),
        )
        torch.save(
            self.target_actor_optimiser.state_dict(),
            pathJoin(save_dir, self.target_actor.optimiser_save_file_name),
        )

    def load(self, save_dir: str):
        self.critic_1.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.critic_1.save_file_name),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
        self.critic_1_optimiser.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.critic_1.optimiser_save_file_name),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )

        self.target_critic_1.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.target_critic_1.save_file_name),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
        self.target_critic_1_optimiser.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.target_critic_1.optimiser_save_file_name),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )

        self.critic_2.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.critic_2.save_file_name),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
        self.critic_2_optimiser.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.critic_2.optimiser_save_file_name),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )

        self.target_critic_2.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.target_critic_2.save_file_name),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
        self.target_critic_2_optimiser.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.target_critic_2.optimiser_save_file_name),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )

        self.actor.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.actor.save_file_name),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
        self.actor_optimiser.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.actor.optimiser_save_file_name),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )

        self.target_actor.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.target_actor.save_file_name),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
        self.target_actor_optimiser.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.target_actor.optimiser_save_file_name),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
