import torch
import numpy as np


class ReplayBuffer:
    """
    Creates a memory buffer for storing experiences

    """

    def __init__(self, batchSize):
        self.states = []
        self.probabilities = []
        self.criticValues = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batchSize = batchSize

    def generateBatches(self):
        numStates = len(self.states)
        batchStart = np.arange(0, numStates, self.batchSize)
        indices = np.arange(numStates, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batchSize] for i in batchStart]
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probabilities),
            np.array(self.criticValues),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def store(self, state, action, probabilities, valuations, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probabilities.append(probabilities)
        self.criticValues.append(valuations)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.probabilities = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.criticValues = []


import torch
import numpy as np


class ReplayBufferTwo:
    """
    Creates a memory buffer for storing experiences

    """

    def __init__(self, max_size: int, state_n: int, actions_n: int):
        self.max_size = max_size
        self.state_memory = torch.zeros(
            (self.max_size, actions_n, state_n), dtype=torch.float32
        )
        self.new_state_memory = torch.zeros(
            (self.max_size, actions_n, state_n), dtype=torch.float32
        )
        self.action_memory = torch.zeros(
            (self.max_size, actions_n), dtype=torch.float32
        )
        self.reward_memory = torch.zeros(self.max_size, dtype=torch.float32)
        self.done_memory = torch.zeros(self.max_size, dtype=torch.bool)
        self.memory_full = False
        self.usePriority = True
        self.pointer = 0

    def genDist(self):
        # A linear probability distribution. Sample selection is
        # proportional to recency
        if self.memory_full:
            amount = np.arange(self.max_size)
            pDist = (
                np.concatenate(
                    (amount[self.max_size - self.pointer :], amount[: self.pointer])
                )
                + 1
            ) / (0.5 * self.max_size * (self.max_size + 1))
        else:
            pDist = (np.arange(self.pointer) + 1) / (
                0.5 * self.pointer * (self.pointer + 1)
            )
        return pDist

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: np.float32,
        new_state: torch.Tensor,
        done: torch.BoolType,
    ):
        self.state_memory[self.pointer] = state
        self.new_state_memory[self.pointer] = new_state
        self.action_memory[self.pointer] = action
        self.reward_memory[self.pointer] = reward
        self.done_memory[self.pointer] = done

        self.pointer += 1
        if self.pointer >= self.max_size:
            self.pointer = 0
            self.memory_full = True

    def sample(self, batch_size, use_pointer=False):
        batch = np.random.choice(
            self.max_size if not use_pointer else self.pointer,
            batch_size,
            p=self.genDist() if self.usePriority else None,
        )
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        done_arr = self.done_memory[batch]

        return states, actions, rewards, new_states, done_arr
