import numpy as np
import pandas as pd

from datetime import datetime
from scipy.special import softmax

def dataset_split(size: int, min_size: int, validation_split: float,
    test_split: float) -> np.ndarray:
    assert (test_split + validation_split) < 0.5

    validation_size = max(int(validation_split * size), min_size)
    test_size = max(int(test_split * size), min_size)
    
    return np.array([
        0, size - validation_size - test_size, size - test_size, size
    ])

def kfold_split(size: int, max_k: int, min_size: int,
    validation_split: float, test_split: float) -> np.ndarray:
    """ Returns [K x 4] matrix where each entry in K denotes the
        indices bounding the train, validation and test datasets
        for the respective fold.
    """
    dataset_splits = []

    # appended in reverse order
    for _ in range(max_k):
        try:
            dataset_splits.append(
                dataset_split(size, min_size, validation_split, test_split)
            )
        except:
            break

        size = dataset_splits[-1][2]

    return np.stack(dataset_splits[::-1])

class EpisodePipeLine:
    @staticmethod
    def mode(mode_type: str):
        if mode_type == "training" or mode_type == "train":
            return 0
        if mode_type == "validation" or mode_type == "validate":
            return 1
        if mode_type == "testing" or mode_type == "test":
            return 2
        
        raise Exception(f"The mode {mode_type} not recognized.")

    def __init__(self, ticker_returns: pd.DataFrame, ticker_observations: np.ndarray,
        ticker_covariance: np.ndarray, portfolio_size: int, episode_size: int,
        window_size: int, validation_split: float = 0.1, test_split: float = 0.1,
        benchmark_returns: np.ndarray = None, riskfree_component: int = True,
        shuffle_tickers: int = True):
        """ Parameters:
                ticker_returns: pd.DataFrame [T x M]
                    riskfree components must be set as the end column.
                ticker_observations: np.ndarray [T x M x F]
                ticker_covariance: np.ndarray [T x M x M]
                benchmark_returns: np,ndarray [T]
                riskfree_component: int
                    whether the dataset has a riskfree component
        """
        self.portfolio_size = portfolio_size
        self.episode_size = episode_size
        self.window_size = window_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.riskfree_component = riskfree_component
        self.shuffle_tickers = shuffle_tickers

        self.universe_size = ticker_returns.shape[1]
        self.ticker_universe = ticker_returns.columns.values

        assert (ticker_returns.shape[0] == ticker_observations.shape[0] and
                ticker_returns.shape[0] == ticker_covariance.shape[0])

        self.date_range = ticker_returns.index.values
        self.ticker_returns = ticker_returns.values
        self.ticker_observations = ticker_observations
        self.ticker_covariance = ticker_covariance

        if benchmark_returns is None:
            self.benchmark_returns = np.zeros(
                shape=(self.ticker_returns.shape[0],),
                dtype=np.float64
            )
        else:
            assert ticker_returns.shape[0] == benchmark_returns.size
            self.benchmark_returns = benchmark_returns

        # experience counter for each timestep
        self.timestep_exp = np.ones(self.ticker_returns.shape[0], dtype=int)

        # bounds splitting dataset into training, validation and testing
        self.dataset_splits = dataset_split(
            self.ticker_returns.shape[0],
            self.episode_size + self.window_size,
            self.validation_split,
            self.test_split
        )

    def reset_episode(self, mode: int = 0, timestep: int = None, end_timestep: int = None,
        **kwargs):
        """ Resets the episode according to the modes [0, 1, 2]
            corresponding to {training, validation, testing}
        """
        # random selection of portfolio tickers
        if self.riskfree_component:
            self.portfolio_indices = np.append(
                np.random.choice(
                    np.arange(0, self.universe_size - 1),
                    size=(self.portfolio_size - 1),
                    replace=False
                ), [self.universe_size - 1]
            )
        else:
            self.portfolio_indices = np.random.choice(
                np.arange(0, self.universe_size),
                size=self.portfolio_size,
                replace=False
            )
        
        if not self.shuffle_tickers:
            # maintain order of tickers
            self.portfolio_indices = np.sort(self.portfolio_indices)
        
        min_timestep = self.dataset_splits[mode] + self.window_size
        max_timestep = self.dataset_splits[mode + 1] - self.episode_size + 1

        if timestep:
            self.timestep = timestep
        else:
            self.timestep = np.random.choice(
                np.arange(min_timestep, max_timestep),
                p=softmax(1. / self.timestep_exp[min_timestep:max_timestep])
            )

        if end_timestep: # bypasses set checks
            self.end_timestep = end_timestep
        else:
            self.end_timestep = self.timestep + self.episode_size

        self.steps_taken = 0

    def episode_ended(self):
        # self.steps_taken >= self.episode_size
        return self.timestep >= self.end_timestep

    def step_timestamp(self):
        return self.date_range[self.timestep]

    def step_observation(self) -> tuple:
        # returns window of previous observations and previous timestep covariance
        return (
            self.ticker_observations[(self.timestep - self.window_size):self.timestep,
                    self.portfolio_indices],
            self.ticker_covariance[np.ix_([self.timestep - 1], self.portfolio_indices,
                    self.portfolio_indices)][0]
        )

    def step_returns(self) -> tuple:
        # returns the returns for the current step
        return (
            self.ticker_returns[self.timestep, self.portfolio_indices],
            self.benchmark_returns[self.timestep]
        )
    
    def take_step(self):
        # increments the experience counter and timestep
        self.timestep_exp[self.timestep] += 1
        self.timestep += 1
        self.steps_taken += 1

    def timestep_from_date(self, date: str, format: str = "%Y-%m-%d",
        greater_equality: int = True) -> int:

        date = datetime.strptime(date, format).strftime("%Y-%m-%d")
        date = np.datetime64(date + "T00:00:00.000000000")

        if greater_equality:
            try:
                return np.where(self.date_range >= date)[0][0]
            except:
                return self.date_range.size
        try:    
            return np.where(self.date_range <= date)[0][-1]
        except:
            return 0

class KFoldPipeLine(EpisodePipeLine):
    def __init__(self, ticker_returns: pd.DataFrame, ticker_observations: np.ndarray,
        ticker_covariance: np.ndarray, portfolio_size: int, episode_size: int,
        window_size: int, max_k: int, validation_split: float = 0.1, test_split: float = 0.1,
        benchmark_returns: np.ndarray = None, riskfree_component: int = True,
        shuffle_tickers: int = True):

        super().__init__(
            ticker_returns, ticker_observations, ticker_covariance,
            portfolio_size, episode_size, window_size, validation_split,
            test_split, benchmark_returns, riskfree_component, shuffle_tickers
        )

        self.dataset_splits = kfold_split(
            self.ticker_returns.shape[0],
            max_k=max_k,
            min_size=self.episode_size + self.window_size,
            validation_split=self.validation_split,
            test_split=self.test_split
        )

        self.k = 0
        self.max_k = self._dataset_splits.shape[0]
  
    @property
    def dataset_splits(self):
        return self._dataset_splits[self.k]

    @dataset_splits.setter
    def dataset_splits(self, split: np.ndarray):
        self._dataset_splits = split

    def set_k(self, k: int):
        assert k <= self.max_k
        self.k = (k - 1) # convert to zero-indexed

    def set_test_splits(self, k: int, split_start: int, split_end: int):
        self._dataset_splits[k][2] = split_start
        self._dataset_splits[k][3] = split_end

        # adjust train-validation split
        self._dataset_splits[k][1] = int(
            self._dataset_splits[k][2] * (1. - self.validation_split)
        )

    def reset_episode(self, mode: int = 0, k: int = 1, **kwargs):
        self.set_k(k)
        super().reset_episode(mode, **kwargs)

    def split_start_from_date(self, date: str, format: str = "%Y-%m-%d",
        greater_equality: int = True) -> int:

        timestep = self.timestep_from_date(
            date, format, greater_equality
        ) - self.window_size

        assert timestep >= 0
        return timestep

if __name__ == "__main__":
    pass