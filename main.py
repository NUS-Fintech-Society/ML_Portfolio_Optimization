import numpy as np
import pandas as pd
from data_scraping.DataScraper import DataScraper
from data_scraping.default import constituent_list
from matplotlib import plt
from optimizer.Optimizer import PortfolioOptimizer
from optimizer.returns_forecast.LSTM.model import LstmModel
from pypfopt import Plotting
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import risk_matrix
from sklearn.model_selection import train_test_split

data = DataScraper.load_price_data()

train_data, test_data = train_test_split(data, test_size=0.3, shuffle=False)

series_arr = []

for ticker in constituent_list:
    temp_train = train_data[ticker]
    temp_test = test_data[ticker]
    model = LstmModel().fit(train_data=temp_train.values)
    predictions = model.batch_predict(data=temp_test.values)
    series = pd.Series(predictions.reshape([-1]), index=temp_test.index[60:])
    series.name = ticker
    series_arr.append(series)

forecasted_prices = pd.concat(series_arr, axis=1)
forecasted_returns = forecasted_prices.pct_change()[1:]
cov = forecasted_returns.iloc[:30].cov()
plotting.plot_efficient_frontier(
    EfficientFrontier(forecasted_returns.values[60], risk_matrix(forecasted_returns.values[:60], returns_data=True)))
plt.show()
