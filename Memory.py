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
