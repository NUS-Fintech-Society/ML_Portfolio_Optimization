import pickle
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import exp_cov
from scipy.special import logsumexp
from tqdm import tqdm
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils.common import Checkpointer
from tf_agents.agents.ddpg.ddpg_agent import DdpgAgent

from tf_agents.agents.ddpg.critic_network import CriticNetwork

from config import *
from episode_pipeline import EpisodePipeLine
from observers import RewardAccumulator
from networks import ActorNetworkCustom
from trade_environment import LogExcessReturn, LogSumReturn, TradeEnvironment
from preprocessing import moving_pypfopt_apply, ticker_history, preprocess_pipeline

warnings.filterwarnings("ignore")

# config *******************************************************************
def preprocess_observations(ticker_prices: pd.DataFrame,
    ticker_returns: pd.DataFrame, **kwargs) -> np.ndarray:
    
    expected_returns = moving_pypfopt_apply(
        method=ema_historical_return,
        ticker_prices=ticker_prices,
        span=FEATURE_WINDOW
    )

    return np.stack([
        ticker_prices.values[-expected_returns.shape[0]:],
        ticker_returns.values[-expected_returns.shape[0]:],
        expected_returns
    ], axis=-1)

def preprocess_covariance(ticker_prices: pd.DataFrame, **kwargs) -> np.ndarray:

    return moving_pypfopt_apply(
        method=exp_cov,
        ticker_prices=ticker_prices,
        span=FEATURE_WINDOW
    )  

def compile_ddpg_agent(environment: TFEnvironment) -> DdpgAgent:
    actor_network = ActorNetworkCustom(
        input_tensor_spec=environment.observation_spec(),
        output_tensor_spec=environment.action_spec(),
        window_size=WINDOW_SIZE,
        scoring_features=SCORING_FEATURES,
        scoring_conv_filters=SCORING_CONV_FILTERS,
        scoring_kernel_size=SCORING_KERNEL_SIZE,
        secondary_dense_units=SECONDARY_DENSE_UNITS
    )

    critic_network = CriticNetwork(
        input_tensor_spec=(
            environment.observation_spec(),
            environment.action_spec()
        ),
        observation_fc_layer_params=[32, 64],
        action_fc_layer_params=[16, 32],
        joint_fc_layer_params=[128, 64, 32]
    )

    ddpg_agent = DdpgAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=tf.optimizers.Adam(LEARNING_RATE),
        critic_optimizer=tf.optimizers.Adam(LEARNING_RATE),
        ou_stddev=OU_STD_DEV,
        target_update_tau=TARGET_UPDATE_TAU,
        train_step_counter=(
            tf.compat.v1.train.get_or_create_global_step()
        )
    )

    ddpg_agent.initialize()
    return ddpg_agent

# **************************************************************************

def restore_agent_checkpointer(agent, ckpt_dirpath: str) -> Checkpointer:
    checkpointer = Checkpointer(
        ckpt_dir=ckpt_dirpath,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        collect_policy=agent.collect_policy
    )

    checkpointer.initialize_or_restore()
    return checkpointer

def moded_environments(ep_pipeline: EpisodePipeLine, modes,
    **env_kwargs) -> tuple:
    return (
        TradeEnvironment(
            episode_pipeline=ep_pipeline,
            **env_kwargs,
            mode=EpisodePipeLine.mode(mode)
        ) for mode in modes
    )

def train_agent(ep_pipeline: EpisodePipeLine, compile_agent,
    ckpt_dirpath: str, batch_size: int, buffer_capacity,
    prepend_episodes: int, episodes: int, epochs_per_ep: int,
    validation_episodes: int):
    # * currently no benchmark for early stopping

    train_env, validate_env = (
        TFPyEnvironment(env) for env in moded_environments(
            ep_pipeline=ep_pipeline,
            modes=["training", "validation"],
            reward_funct=LogExcessReturn
        )
    )
    
    # compile and restore agent
    agent = compile_agent(environment=train_env)
    checkpointer = restore_agent_checkpointer(agent, ckpt_dirpath)
    
    replay_buffer = TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=buffer_capacity
    )

    dataset = iter(replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=agent.train_sequence_length
    ))

    # training drivers
    append_episode_driver = DynamicEpisodeDriver(
        env=train_env,
        policy=agent.collect_policy,
        observers=[
            replay_buffer.add_batch
        ]
    )
    
    # prepend replay_buffer
    for _ in tqdm(range(prepend_episodes), desc="Prepending Replay Buffer"):
        append_episode_driver.run()

    reward_accumulator = RewardAccumulator(store_steps=False)
    validation_driver = DynamicEpisodeDriver(
        env=validate_env,
        policy=agent.policy,
        observers=[
            reward_accumulator
        ]
    )

    for ep in range(episodes):
        append_episode_driver.run()

        for _ in tqdm(range(epochs_per_ep), desc=f"Ep{ep} Training Agent"):
            experience, _ = next(dataset)
            agent.train(experience)
        
        for _ in tqdm(range(validation_episodes), desc=f"Ep{ep} Running Agent Validation"):
            validation_driver.run()

        print(
            f"Ep{ep} Validation Log Average Reward: ",
            logsumexp(
                np.array(reward_accumulator.rewards) -
                np.math.log(validation_episodes)
            )
        )

        reward_accumulator.reset()

    checkpointer.save(agent.train_step_counter)

def test_agent(ep_pipeline: EpisodePipeLine, compile_agent,
    ckpt_dirpath: str, start: str, end: str) -> pd.Series:

    timestep = ep_pipeline.timestep_from_date(start)
    end_timestep = ep_pipeline.timestep_from_date(end)

    date_range = ep_pipeline.date_range[timestep:end_timestep]

    env = TFPyEnvironment(
        TradeEnvironment(
            episode_pipeline=ep_pipeline,
            reward_funct=LogSumReturn,
            timestep=timestep,
            end_timestep=end_timestep
        )
    )
    
    agent = compile_agent(environment=env)
    _ = restore_agent_checkpointer(agent, ckpt_dirpath)

    accumulator = RewardAccumulator(store_steps=True)

    DynamicEpisodeDriver(
        env=env,
        policy=agent.policy,
        observers=[
            accumulator
        ]
    ).run()

    return pd.Series(
        data=np.exp(np.array(accumulator.rewards[0])),
        index=date_range
    )

def main(skip_preprocessing=False, skip_training=False,
    skip_testing=False):

    # data preprocessing *************************************
    if not skip_preprocessing:
        ticker_prices = pd.read_csv("data\ohlc_tickers.csv")
        ticker_prices = ticker_prices.pivot(
            index="Date",
            columns="ticker",
            values="Adj Close"
        ).sort_index()

        benchmark_prices = ticker_history(
            tickers=["^GSPC"],
            price_type="Adj Close"
        )

        # append riskfree constant component
        ticker_prices["RF_CONST"] = 1.

        episode_pipeline = preprocess_pipeline(
            ticker_prices=ticker_prices,
            benchmark_prices=benchmark_prices,
            observations_funct=preprocess_observations,
            covariance_funct=preprocess_covariance
        )
    else:
        with open(PIPE_FPATH, "rb") as handle:
            episode_pipeline = pickle.load(handle)

    if not skip_training:
        for k in range(K_FOLDS):
            print(
                f"Training Agent Fold {k}\n"
                "================================================="
            )
            
            # override k-split boundaries *************************
            episode_pipeline.set_test_splits(
                k,
                split_start=episode_pipeline.split_start_from_date(
                    TEST_PERIODS[k]
                ),
                split_end=episode_pipeline.timestep_from_date(
                    TEST_PERIODS[k], greater_equality=False
                )
            )
            # *****************************************************
            episode_pipeline.set_k(k)

            train_agent(
                ep_pipeline=episode_pipeline,
                compile_agent=compile_ddpg_agent,
                ckpt_dirpath=CKPT_DIRPATH + f"\Agent{k}",
                batch_size=BATCH_SIZE,
                buffer_capacity=BUFFER_CAPACITY,
                prepend_episodes=PREPEND_EPISODES,
                episodes=EPISODES,
                epochs_per_ep=EPOCHS_PER_EP,
                validation_episodes=VALIDATION_EPISODES
            )

    if not skip_testing:
        test_rewards = []

        for k in tqdm(range(K_FOLDS), desc="testing agent"):
            episode_pipeline.set_k(k)

            rewards = test_agent(
                ep_pipeline=episode_pipeline,
                compile_agent=compile_ddpg_agent,
                ckpt_dirpath=CKPT_DIRPATH + f"\Agent{k}",
                start=TEST_PERIODS[k],
                end=TEST_PERIODS[k + 1]
            )
            
            rewards.name = f"FOLD_{k}"
            test_rewards.append(rewards)

        test_rewards = pd.DataFrame(data=test_rewards).transpose()
        print(test_rewards)
        test_rewards.to_csv(r"Saved\test_rewards.csv")

if __name__ == "__main__":
    main(
        skip_preprocessing=True,
        skip_training=True
    )