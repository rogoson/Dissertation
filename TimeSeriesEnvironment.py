import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeSeriesEnvironment(gym.Env):
    def __init__(
        self,
        marketData,
        normData,
        TIME_WINDOW,
        EPISODE_LENGTH,
        startCash,
        AGENT_RISK_AVERSION,  # i know, forgive me
        transactionCost=0.001,
    ):

        self.marketData = marketData
        self.normData = normData
        self.TIME_WINDOW = TIME_WINDOW
        self.timeStep = 0
        self.episodeLength = EPISODE_LENGTH
        self.transactionCost = transactionCost
        self.allocations = []  # For transaction Cost Calculation
        self.startCash = startCash
        self.previousPortfolioValue = None
        self.RETURNS = [0]
        self.PORTFOLIO_VALUES = [self.startCash]
        self.AGENT_RISK_AVERSION = AGENT_RISK_AVERSION
        self.isReady = False

        self.CVaR = [0]
        self.maxAllocationChange = 1  # liquidigy parameter.

        # Required for Differential Sharpe Ratio
        self.decayRate = 0.01
        self.meanReturn = None
        self.meanSquaredReturn = None

    def getData(self, timeStep=None):
        if timeStep is None:
            timeStep = self.timeStep
        data = self.getMarketData(
            self.normData,
            timeStep,
            self.TIME_WINDOW,
        )
        return data

    def getMarketData(self, dataframes, i, TIME_WINDOW):
        relevantData = []
        for p in range(0, TIME_WINDOW):
            rel = np.array(np.append(self.allocations[-p], self.PORTFOLIO_VALUES[-p]))
            for _, data in dataframes.items():
                index = i - p
                toBeAppended = data.iloc[index, :]
                rel = np.append(rel, toBeAppended.values[:-1])  # no need for return
            relevantData.append(rel)
        return np.array(relevantData)

    def step(self, action, rewardMethod="CVaR"):
        reward = (
            self.calculatePortfolioValue(
                action,
                self.marketData.iloc[self.timeStep + 1],
            )
            - self.previousPortfolioValue
        )

        self.timeStep += 1
        info = dict()

        done = False

        # below not really needed if using indexes [virtually impossible to lose all your money]
        if (self.previousPortfolioValue + reward) / self.startCash < 0.7:
            done = True
            info["reason"] = "portfolio_below_70%"
        elif self.timeStep + 1 == self.episodeLength:
            done = True
            info["reason"] = "max_steps_reached"

        self.updateReturns(reward)
        if rewardMethod == "CVaR":
            reward = self.getCVaRReward(reward)
        elif rewardMethod == "Standard Logarithmic Returns":
            reward = self.getCVaRReward(reward, False)
        else:
            reward = self.calculateDifferentialSharpeRatio(reward)

        if info.get("reason") == "portfolio_below_70%":
            reward -= 100 * (reward)  # big penalty for loss of 30%

        return (
            None,
            reward,
            done,
            False,
            info,
        )

    def getMetrics(self, portfolioValues=None, returns=None):
        if portfolioValues == None:
            portfolioValues = self.PORTFOLIO_VALUES
        if returns == None:
            returns = self.RETURNS
        info = dict()
        info["Cumulative \nReturn (%)"] = round(
            100 * (portfolioValues[-1] / self.startCash) - 100, 2
        )
        info["Maximum \nPullback (%)"] = self.maxPullback()
        info["Sharpe Ratio"] = round(np.mean(returns) / np.std(returns), 4)
        info["Total Timesteps"] = self.timeStep
        return info

    def maxPullback(self):
        maxValue = float("-inf")
        maxDrawdown = 0.0
        for value in self.PORTFOLIO_VALUES:
            maxValue = max(maxValue, value)
            drawdown = (maxValue - value) / maxValue * 100
            maxDrawdown = max(maxDrawdown, drawdown)
        return maxDrawdown

    def calculateDifferentialSharpeRatio(self, currentReturn):
        """
        In line with Moody & Saffel's "Reinforcement Learning for Trading" 1998 Paper
        """
        if self.meanReturn is None:
            self.meanReturn = currentReturn
            self.meanSquaredReturn = currentReturn**2
            return 0.0

        prevMeanReturn = self.meanReturn
        prevMeanSquaredReturn = self.meanSquaredReturn

        deltaMean = currentReturn - prevMeanReturn
        deltaSquared = currentReturn**2 - prevMeanSquaredReturn
        self.meanReturn += self.decayRate * deltaMean
        self.meanSquaredReturn += self.decayRate * deltaSquared

        denom = (prevMeanSquaredReturn - prevMeanReturn**2) ** 1.5
        if denom == 0:
            return 0.0
        numerator = (
            prevMeanSquaredReturn * deltaMean - 0.5 * prevMeanReturn * deltaSquared
        )
        differentialSharpeRatio = numerator / denom
        return differentialSharpeRatio

    def normaliseValue(self, value):
        return np.sign(value) * (np.log1p(np.abs(value)))

    def getCVaRReward(self, r, useCVaR=True):
        if useCVaR:
            currentCVaR = self.calculateCVaR()
            changeInCVaR = currentCVaR - self.CVaR[-1]
            cVaRNum = self.normaliseValue(changeInCVaR)
            riskPenalty = self.AGENT_RISK_AVERSION * cVaRNum
            self.CVaR.append(currentCVaR)
        else:
            riskPenalty = 0
        scaledReward = self.normaliseValue(r)
        finalReward = scaledReward - riskPenalty
        return finalReward

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)
        self.timeStep = 0
        self.allocations = []
        self.previousPortfolioValue = None
        self.PORTFOLIO_VALUES = [self.startCash]
        self.isReady = False
        self.CVaR = [0]
        self.RETURNS = [0]
        self.meanReturn = None
        self.meanSquaredReturn = None
        self.startHavingEffect = 0

    def calculatePortfolioValue(self, targetAllocation, closingPriceChanges):
        targetAllocation = np.array(targetAllocation)
        if self.previousPortfolioValue is None:
            self.previousPortfolioValue = self.startCash

        if self.allocations:
            prevAllocation = np.array(self.allocations[-1])
            currentAllocation = (
                1 - self.maxAllocationChange
            ) * prevAllocation + self.maxAllocationChange * targetAllocation
        else:
            prevAllocation = np.zeros(len(closingPriceChanges) + 1)
            prevAllocation[0] = 1
            self.allocations.append(prevAllocation)
            currentAllocation = targetAllocation
        wealthDistribution = self.previousPortfolioValue * currentAllocation
        # 1 for cash - presumed not to change
        closingPriceChanges = np.insert(closingPriceChanges, 0, 0)
        changeWealth = (1 + closingPriceChanges) * wealthDistribution

        transactionCost = 0
        if self.transactionCost > 0:
            transactionCost = self.transactionCost * np.sum(
                self.previousPortfolioValue
                * np.abs(currentAllocation - np.array(self.allocations[-1]))
            )
        self.allocations.append(currentAllocation)
        return np.sum(changeWealth) - transactionCost

    def getPrices(
        self,
    ):
        prices = []
        for product in self.marketData.values():
            prices.append(
                product["close"],
            )
        return prices

    def calculateCVaR(self, percentage=0.05):
        if len(self.CVaR) < max(10, int(1 / percentage)):
            return np.mean(self.CVaR)
        sortedReturns = sorted(self.RETURNS)
        indexToBePicked = int(np.ceil(percentage * len(sortedReturns)))
        CVaR = np.mean(sortedReturns[:indexToBePicked])
        return CVaR

    def updateReturns(self, profit):
        self.RETURNS.append(profit)
        self.PORTFOLIO_VALUES.append(
            self.previousPortfolioValue + profit
        )  # already updated earlier
        self.previousPortfolioValue = self.previousPortfolioValue + profit

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def getIsReady(self):
        return self.isReady

    def setIsReady(self, boolean: bool):
        self.isReady = boolean
