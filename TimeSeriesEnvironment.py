import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces


class TimeSeriesEnvironment(gym.Env):
    def __init__(
        self,
        marketData,
        START_INDEX,
        TIME_WINDOW,
        TRAINING_EPS,
        startCash,
        AGENT_RISK_AVERSION,  # i know, forgive me
        transactionCost=0.001,
    ):

        self.marketData = marketData
        self.START_INDEX = START_INDEX
        self.TIME_WINDOW = TIME_WINDOW
        self.timeStep = 0
        self.TRAINING_EPS = TRAINING_EPS
        self.transactionCost = transactionCost
        self.allocations = []  # For transaction Cost Calculation
        self.startCash = startCash
        self.previousPortfolioValue = None
        self.RETURNS = [0]
        self.PORTFOLIO_VALUES = [self.startCash]
        self.AGENT_RISK_AVERSION = AGENT_RISK_AVERSION
        self.isReady = False

        self.CVaR = [0]
        self.traded = 0  # counts number of trades, for APPT
        self.startHavingEffect = 0  # essentially a one-time switch
        self.countsIndex = 0
        self.turbulenceThreshold = self.getTurbulenceThreshold()
        self.maxAllocationChange = 0.5

        # Required for Differential Sharpe Ratio
        self.decayRate = 0.01
        self.meanReturn = None
        self.meanSquaredReturn = None

    def getTurbulenceThreshold(self, endWindow=None):
        """
        Based on formula found here:
        https://portfoliooptimizer.io/blog/the-turbulence-index-regime-based-partitioning-of-asset-returns/
        """
        if endWindow == None:
            endWindow = int(self.TRAINING_EPS * 2 / 3)
        returns = []
        turbList = []
        for _, frame in self.marketData.items():
            returns.append(frame["Return"].values)
        returns = np.array(returns).T
        for i in range(self.TIME_WINDOW, endWindow):
            historicalReturns = returns[
                self.START_INDEX + i - self.TIME_WINDOW : i + self.START_INDEX, :
            ]
            mean = np.mean(historicalReturns, axis=0)
            covariance = np.cov(historicalReturns, rowvar=False)
            try:
                inverseCovariance = np.linalg.inv(covariance)
            except np.linalg.LinAlgError:
                inverseCovariance = np.linalg.pinv(covariance)
            deviation = returns[i] - mean
            turbulence = deviation.T @ inverseCovariance @ deviation
            turbList.append(turbulence)
        threshold = np.percentile(turbList, 90)
        return threshold

    def getCurrentTurbulence(self):
        if self.timeStep < self.TIME_WINDOW:
            return 0
        historicalReturns = []
        for _, frame in self.marketData.items():
            historicalReturns.append(
                frame["Return"].values[
                    self.START_INDEX
                    + self.timeStep
                    - self.TIME_WINDOW : self.START_INDEX
                    + self.timeStep
                ]
            )
        historicalReturns = np.array(historicalReturns).T
        mean = np.mean(historicalReturns, axis=0)
        covariance = np.cov(historicalReturns, rowvar=False)
        currentReturn = historicalReturns[-1, :]
        try:
            inverseCovariance = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            inverseCovariance = np.linalg.pinv(covariance)
        deviation = currentReturn - mean
        turbulence = deviation.T @ inverseCovariance @ deviation
        return turbulence

    def getData(self, timeStep=None):
        if timeStep is None:
            timeStep = self.timeStep
        data = self.getMarketData(
            self.marketData,
            self.START_INDEX,
            timeStep,
            self.TIME_WINDOW,
        )
        return data  # not including the (to be predicted) last timestep

    def getMarketData(self, dataframes, START_INDEX, i, TIME_WINDOW):
        relevantData = []
        for p in range(0, TIME_WINDOW):
            rel = np.array(
                np.append(self.allocations[-p][1:], self.PORTFOLIO_VALUES[-p])
            )
            for _, data in dataframes.items():
                index = START_INDEX + i - p
                toBeAppended = data.iloc[index, :]
                rel = np.append(rel, toBeAppended.values[:-1])  # no need for return
            relevantData.append(rel)
        return np.array(relevantData)

    def getChangeData(self, dataframes, START_INDEX, i):
        relevantData = []
        for _, data in dataframes.items():
            firstIndex = START_INDEX + i
            relevantData.append(
                data["close"].iloc[firstIndex : firstIndex + 2]
            )  # skip time column
        return relevantData

    def step(self, action, haveEffect, rewardMethod="CVaR"):
        relevantData = self.getChangeData(
            self.marketData, self.START_INDEX, self.timeStep
        )
        reward = (
            self.returnNewPortfolioValue(
                self.marketData.keys(),
                relevantData,
                action,
            )
            - self.previousPortfolioValue
        )

        self.timeStep += 1
        info = dict()

        # turbulence: if above threshold, alert agent
        turbulence = self.getCurrentTurbulence()
        if turbulence > self.turbulenceThreshold:
            info["turbulence_breached"] = True
        else:
            info["turbulence_breached"] = False
        terminated = False
        done = False

        # below not really needed if using indexes [virtually impossible to lose all your money]
        if (self.previousPortfolioValue + reward) / self.startCash < 0.7:
            done = True
            info["reason"] = "portfolio_below_70%"
        elif self.timeStep == self.TRAINING_EPS:
            done = True
            info["reason"] = "max_steps_reached"

        if haveEffect:
            self.startHavingEffect += 1
            self.turbulenceThreshold = self.getTurbulenceThreshold(self.timeStep)
            if self.startHavingEffect == 1:  # forgive me
                self.countsIndex = self.timeStep
                self.previousPortfolioValue = self.startCash
                self.traded = 0
                self.meanReturn = None
        self.updateReturns(reward)

        if rewardMethod == "CVaR":
            reward = self.getCVaRReward(reward)
        elif rewardMethod == "Standard Log":
            reward = self.getCVaRReward(reward, False)
        else:
            reward = self.calculateDifferentialSharpeRatio(reward)
        return (
            (
                self.getData() if self.isReady else None
            ),  # little weird but this now returns next timestep data
            reward,
            done,
            False,
            info,
        )

    def getMetrics(self, portfolioValues=None, returns=None):
        if portfolioValues == None:
            portfolioValues = self.PORTFOLIO_VALUES[self.countsIndex :]
        if returns == None:
            returns = self.RETURNS[self.countsIndex :]
        info = dict()
        info["Cumulative \nReturn (%)"] = round(
            100 * (portfolioValues[-1] / self.startCash) - 100, 2
        )
        info["Maximum Earning \nRate (%)"] = round(
            100 * (np.max(portfolioValues) / self.startCash) - 100, 2
        )
        info["Maximum \nPullback (%)"] = self.maxPullback(portfolioValues)
        info["Average Profitability \nper Trade"] = (
            portfolioValues[-1] - self.startCash
        ) / self.traded
        info["Sharpe Ratio"] = round(np.mean(returns) / np.std(returns), 4)
        info["Total Timesteps"] = self.timeStep
        return info

    def maxPullback(self, portfolioValues):
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
        self.traded = 0
        self.meanReturn = None
        self.meanSquaredReturn = None
        self.startHavingEffect = 0
        self.countsIndex = 0

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
            currentAllocation = targetAllocation
        wealthDistribution = self.previousPortfolioValue * currentAllocation
        # 1 for cash - presumed not to change
        changeWealth = (
            np.array([1] + list(closingPriceChanges.values())) * wealthDistribution
        )

        transactionCost = 0
        if self.allocations:
            transactionCost = self.transactionCost * np.sum(
                self.previousPortfolioValue
                * np.abs(currentAllocation - np.array(self.allocations[-1]))
            )
        self.allocations.append(currentAllocation.tolist())
        return np.sum(changeWealth) - transactionCost

    def returnNewPortfolioValue(self, dataKeys, relevantData, allocation):
        priceChanges = dict()
        for index, product in enumerate(dataKeys):
            priceChanges[product] = relevantData[index].iloc[-1] / (
                relevantData[index].iloc[-2]
            )
        return self.calculatePortfolioValue(allocation, priceChanges)

    def getPrices(
        self,
    ):
        prices = []
        for product in self.marketData.values():
            prices.append(
                [product["close"].iloc[: self.TRAINING_EPS]],
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
