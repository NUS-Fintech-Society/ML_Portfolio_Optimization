import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class LstmModel:
    # Period is the number of windows before that the model considers to predict next data point
    def __init__(self, period: int = 60):
        self.period = period
        self.model = None
        self.scaler = None

    def fit(self, train_data: np.array):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = self.scaler.fit_transform(train_data.reshape(-1, 1))
        x_train = []
        y_train = []
        for i in range(self.period, len(train_data)):
            x_train.append(scaled_train[i - self.period:i, 0])
            y_train.append(scaled_train[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=25, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=12, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='relu'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=15, batch_size=128)
        self.model = model
        return self

    def test(self, test_data: np.array):
        predictions = self.batch_predict(test_data)
        mse = mean_squared_error(test_data[self.period:], predictions)
        rmse = mean_squared_error(test_data[self.period:], predictions, squared=False)
        return mse, rmse

    def batch_predict(self, data: np.array):
        assert self.model is not None, "Model has not been trained, please run model.fit() first"
        assert len(data) >= self.period, "Data passed into function does not match input length required"

        x_data = []
        y_data = []
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        for i in range(self.period, len(data)):
            x_data.append(scaled_data[i - self.period:i, 0])
            y_data.append(scaled_data[i, 0])
        x_test = np.array(x_data)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predicted = self.model.predict(x_test)
        predictions = self.scaler.inverse_transform(predicted)
        print(f"MSE: {mean_squared_error(data[60:], predictions)}")
        print(f"RMSE: {mean_squared_error(data[60:], predictions, squared = False)}")
        return predictions
