import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class PortfolioOptimizer:
    def __init__(self):
        self.p_ret = None
        self.p_vol = None
        self.p_weights = None

    def fit(self, asset_returns: np.ndarray, asset_covariance: np.ndarray, num_iterations: int = 1000):
        # Make sure that input data is of correct format
        n = len(asset_returns)
        assert asset_covariance.shape == (n, n), "Invalid input data"

        p_ret = []  # Define an empty array for portfolio returns
        p_vol = []  # Define an empty array for portfolio volatility
        p_weights = []  # Define an empty array for asset weights

        for portfolio in range(num_iterations):
            weights = np.random.random(n)
            weights = weights / np.sum(weights)
            p_weights.append(weights)

            returns = np.dot(weights, asset_returns)
            p_ret.append(returns)

            var = np.dot(weights.T, np.dot(asset_covariance, weights))
            vol = np.sqrt(var)  # Daily standard deviation
            p_vol.append(vol)
        self.p_ret = p_ret
        self.p_vol = p_vol
        self.p_weights = p_weights
        return self

    def min_volatility(self):
        assert self.p_ret is not None, "Model has not been trained, please run model.fit() first"

        index = np.argmin(self.p_vol)
        return self.p_ret[index], self.p_vol[index], self.p_weights[index]

    def max_sharpe_ratio(self, rrf: float):
        assert self.p_ret is not None, "Model has not been trained, please run model.fit() first"

        index = np.argmax([(i - rrf) / j for i, j in zip(self.p_ret, self.p_vol)])
        return self.p_ret[index], self.p_vol[index], self.p_weights[index]

    def plot_efficient_frontier(self):
        sns.scatterplot(x=self.p_vol, y=self.p_ret)
        plt.show()
