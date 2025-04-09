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


class ReplayBufferTwo:
    def __init__(self, maxSize):
        self.maxSize = maxSize
        self.stateMemory = []
        self.newStateMemory = []
        self.actionMemory = []
        self.rewardMemory = []
        self.doneMemory = []
        self.memoryFull = False
        self.usePriority = True
        self.pointer = 0

    def genDist(self):
        if self.memoryFull:
            amount = np.arange(self.maxSize)
            reordered = np.concatenate(
                (amount[self.maxSize - self.pointer :], amount[: self.pointer])
            )
            pDist = (reordered + 1) / (0.5 * self.maxSize * (self.maxSize + 1))
        else:
            pDist = (np.arange(self.pointer) + 1) / (
                0.5 * self.pointer * (self.pointer + 1)
            )
        return pDist

    def store(self, state, action, reward, newState, done):
        if len(self.stateMemory) < self.maxSize:
            self.stateMemory.append(state)
            self.actionMemory.append(action)
            self.rewardMemory.append(reward)
            self.newStateMemory.append(newState)
            self.doneMemory.append(done)
        else:
            self.stateMemory[self.pointer] = state
            self.actionMemory[self.pointer] = action
            self.rewardMemory[self.pointer] = reward
            self.newStateMemory[self.pointer] = newState
            self.doneMemory[self.pointer] = done

        self.pointer += 1
        if self.pointer >= self.maxSize:
            self.pointer = 0
            self.memoryFull = True

    def sample(self, batchSize, usePointer=False):
        size = self.maxSize if not usePointer else self.pointer
        pDist = self.genDist() if self.usePriority else None
        batch = np.random.choice(size, batchSize, p=pDist)
        states = np.array(self.stateMemory)[batch]
        actions = np.array(self.actionMemory)[batch]
        rewards = np.array(self.rewardMemory)[batch]
        newStates = np.array(self.newStateMemory)[batch]
        doneArr = np.array(self.doneMemory)[batch]
        return states, actions, rewards, newStates, doneArr

    def clear(self):
        self.stateMemory = []
        self.actionMemory = []
        self.rewardMemory = []
        self.newStateMemory = []
        self.doneMemory = []
        self.memoryFull = False
        self.pointer = 0
