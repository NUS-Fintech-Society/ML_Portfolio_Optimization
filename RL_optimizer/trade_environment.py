import numpy as np

from pypfopt.efficient_frontier import EfficientFrontier
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step
from tf_agents.specs.array_spec import ArraySpec 
from tf_agents.specs.array_spec import BoundedArraySpec

from episode_pipeline import EpisodePipeLine
from networks import ActorNetworkCustom

class PercentageCost:
    def __init__(self, percentage_cost: float):
        self.percentage_cost = percentage_cost
    
    def __call__(self, prior_weights: np.ndarray, new_weights: np.ndarray):
        return np.squeeze(new_weights) * (1. - self.percentage_cost * 
            np.sum(np.abs(new_weights - prior_weights)))

def LogSumReturn(portfolio_positions: np.ndarray, **kwargs) -> float:
    return np.math.log(np.sum(portfolio_positions))

def LogExcessReturn(portfolio_positions: np.ndarray, benchmark_return: float,
    **kwargs) -> float:
    return LogSumReturn(portfolio_positions) - LogSumReturn(benchmark_return)

class TradeEnvironment(PyEnvironment):
    def __init__(self, episode_pipeline: EpisodePipeLine, discount: float = 1.,
        preset_weights: np.ndarray = None, transition_funct = PercentageCost(0.001),
        reward_funct = LogSumReturn, **reset_kwargs):
        
        self.episode_pipeline = episode_pipeline
        self.discount = discount
        self.preset_weights = preset_weights
        self.transition_funct = transition_funct
        self.reward_funct = reward_funct
        self.reset_kwargs = reset_kwargs

        self._action_spec = BoundedArraySpec(
            shape=(self.episode_pipeline.portfolio_size,),
            dtype=np.float64,
            minimum=0.,
            maximum=1.
        )

        self._reset()
        self._observation_spec = ArraySpec(
            shape=self.step_observation().shape,
            dtype=np.float64
        )

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def step_observation(self) -> np.ndarray:
        observation, covariance = self.episode_pipeline.step_observation()

        # custom preprocessing of observations ***************
        expected_returns = observation[-1, :, -1]
        
        try:
            tangent_weights = EfficientFrontier(
                expected_returns=expected_returns,
                cov_matrix=covariance
            ).max_sharpe()

            self.tangent_weights = np.array([
                tangent_weights[ticker] for ticker
                in range(self.episode_pipeline.portfolio_size)
            ])
        except: # set as previous tangent weights
            pass

        return ActorNetworkCustom.compat_observation(
            observation,
            np.concatenate([
                covariance,
                np.expand_dims(self.tangent_weights, axis=0),
                np.expand_dims(self.portfolio_weights, axis=0)
            ], axis=0)
        )
        # ****************************************************

    def _reset(self):
        self.episode_pipeline.reset_episode(**self.reset_kwargs)

        if self.preset_weights is None:
            self.portfolio_weights = np.zeros(
                shape=(self.episode_pipeline.portfolio_size,),
                dtype=np.float64
            )

            self.portfolio_weights[-1] = 1.
        else:
            self.portfolio_weights = self.preset_weights

        # custom
        self.tangent_weights = self.portfolio_weights
        
        return time_step.restart(observation=self.step_observation())

    def _step(self, action: np.ndarray):
        if self.episode_pipeline.episode_ended():
            # prior termination step did not actually terminate (bug)
            return self._reset()

        portfolio_returns, benchmark_return = self.episode_pipeline.step_returns()

        # enact transition costs
        portfolio_positions = self.transition_funct(
            self.portfolio_weights, action
        )

        portfolio_positions *= portfolio_returns

        reward = self.reward_funct(
            portfolio_positions=portfolio_positions,
            portfolio_returns=portfolio_returns,
            benchmark_return=benchmark_return
        )

        # re-standardise
        self.portfolio_weights = (portfolio_positions /
                np.sum(portfolio_positions))

        self.episode_pipeline.take_step()

        if self.episode_pipeline.episode_ended():
            return time_step.termination(
                observation=self.step_observation(),
                reward=reward
            )

        return time_step.transition(
            observation=self.step_observation(),
            reward=reward,
            discount=self.discount
        )

if __name__ == "__main__":
    pass