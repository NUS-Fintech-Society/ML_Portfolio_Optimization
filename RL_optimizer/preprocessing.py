import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config import *
from episode_pipeline import EpisodePipeLine, KFoldPipeLine

# utility *************************************************************
def ticker_history(tickers, price_type: str = "Close", start = None,
    end = None) -> pd.DataFrame:
    # wrapper for yfinance download
    ticker_prices = yf.download(
        tickers=[*tickers], start=start, end=end
    )[price_type]

    if isinstance(ticker_prices, pd.Series):
        ticker_prices = ticker_prices.to_frame(name=tickers[0])

    return ticker_prices[tickers]

def max_contiguous_subframe(df: pd.DataFrame, predicate) -> tuple:
    # returns the bounds for the contiguous subset of maximum size
    subset_bounds = np.concatenate([
        [-1],
        np.arange(0, df.shape[0])[~predicate],
        [df.shape[0]]
    ])

    max_subset = (subset_bounds[1:] - subset_bounds[:-1]).argmax()
    return df[
        (subset_bounds[max_subset] + 1):subset_bounds[max_subset + 1]
    ]

def moving_pypfopt_apply(method, ticker_prices: pd.DataFrame,
    span: int = 180) -> np.ndarray:
    """ Returns:
            - expected_returns: np.ndarray [T x M]
            - covariance: np.ndarray [T x M x M]
    """
    return np.stack([
        method(ticker_prices[ts:ts + span], span=span).values
        for ts in tqdm(
            range(ticker_prices.shape[0] - span + 1),
            desc="moving_pypfopt_apply progress:"
        )
    ])

# *********************************************************************
def preprocess_pipeline(ticker_prices: pd.DataFrame, benchmark_prices: pd.DataFrame,
    observations_funct, covariance_funct) -> EpisodePipeLine:

    merged_prices = ticker_prices.join(benchmark_prices)

    # truncate to maximum contiguous subset
    merged_prices = max_contiguous_subframe(
        merged_prices, merged_prices.notnull().all(axis=1)
    )

    ticker_prices = merged_prices[ticker_prices.columns]
    benchmark_prices = merged_prices[benchmark_prices.columns]

    ticker_returns = pd.DataFrame(
        data=ticker_prices.values[1:] / ticker_prices.values[:-1],
        index=ticker_prices.index[1:],
        columns=ticker_prices.columns
    )

    benchmark_returns = (benchmark_prices.values[1:] /
            benchmark_prices.values[:-1])

    ticker_observations = observations_funct(
        ticker_prices=ticker_prices,
        ticker_returns=ticker_returns,
        benchmark_prices=benchmark_prices,
        benchmark_returns=benchmark_returns
    )

    ticker_covariance = covariance_funct(
        ticker_prices=ticker_prices,
        ticker_returns=ticker_returns
    )

    date_range_size = min(
        ticker_returns.shape[0],
        ticker_observations.shape[0],
        ticker_covariance.shape[0]
    )

    # truncate datasets to match date ranges
    ticker_prices = ticker_prices[-date_range_size:]
    ticker_returns = ticker_returns[-date_range_size:]
    benchmark_returns = benchmark_returns[-date_range_size:]
    ticker_observations = ticker_observations[-date_range_size:]
    ticker_covariance = ticker_covariance[-date_range_size:]

    episode_pipeline = KFoldPipeLine(
        ticker_returns=ticker_returns,
        ticker_observations=ticker_observations,
        ticker_covariance=ticker_covariance,
        portfolio_size=PORTFOLIO_SIZE,
        episode_size=EPISODE_SIZE,
        window_size=WINDOW_SIZE,
        max_k=K_FOLDS,
        validation_split=VALIDATION_SPLIT,
        test_split=TEST_SPLIT,
        benchmark_returns=benchmark_returns,
        riskfree_component=True,
        shuffle_tickers=SHUFFLE_TICKERS
    )

    # save pipeline
    with open(PIPE_FPATH, "wb") as handle:
        pickle.dump(
            episode_pipeline, handle,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    return episode_pipeline

if __name__ == "__main__":
    pass