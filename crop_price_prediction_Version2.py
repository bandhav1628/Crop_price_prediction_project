import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class CropPricePrediction:

    def __init__(self, price_path, weather_path):
        self.df = self.load_and_merge(price_path, weather_path)

    def load_and_merge(self, price_path, weather_path):
        price_df = pd.read_csv(price_path)
        weather_df = pd.read_csv(weather_path)
        df = pd.merge(price_df, weather_df, on=['date', 'location'])
        df.fillna(method='ffill', inplace=True)
        df['price_lag1'] = df['price'].shift(1)
        df.dropna(inplace=True)
        return df

    # ARIMA Model
    def train_arima(self, order=(5,1,0), steps=30):
        ts = self.df['price']
        model = ARIMA(ts, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast

    # Prophet Model
    def train_prophet(self, periods=30):
        prophet_df = self.df[['date', 'price']].rename(columns={'date': 'ds', 'price': 'y'})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Random Forest and XGBoost
    def train_rf_xgb(self, features, target='price', test_size=0.2, model_type='rf'):
        X = self.df[features]
        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        if model_type == 'rf':
            model = RandomForestRegressor()
        else:
            model = XGBRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_test, y_pred, model

    # LSTM Model
    def prepare_lstm_data(self, features, target='price', sequence_length=10):
        X, y = [], []
        for i in range(len(self.df) - sequence_length):
            X.append(self.df[features].iloc[i:i+sequence_length].values)
            y.append(self.df[target].iloc[i+sequence_length])
        X, y = np.array(X), np.array(y)
        split = int(0.8 * len(X))
        return X[:split], X[split:], y[:split], y[split:]

    def train_lstm(self, X_train, y_train, X_test, y_test, feature_len, sequence_length=10, epochs=10):
        model = Sequential([
            LSTM(50, input_shape=(sequence_length, feature_len)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        y_pred = model.predict(X_test)
        return y_test, y_pred, model

    # Evaluation
    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        return mae, rmse

    # Visualization
    def plot_predictions(self, y_true, y_pred, title='Actual vs Predicted Price'):
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

# Example usage:
if __name__ == "__main__":
    price_path = 'data/market_prices.csv'
    weather_path = 'data/weather_data.csv'
    features = ['weather_temp', 'weather_rain', 'price_lag1']

    cpp = CropPricePrediction(price_path, weather_path)

    # ARIMA
    arima_forecast = cpp.train_arima()
    print("ARIMA Forecast:\n", arima_forecast)

    # Prophet
    prophet_forecast = cpp.train_prophet()
    print("Prophet Forecast:\n", prophet_forecast.tail())

    # Random Forest
    y_test_rf, y_pred_rf, rf_model = cpp.train_rf_xgb(features, model_type='rf')
    cpp.evaluate(y_test_rf, y_pred_rf)
    cpp.plot_predictions(y_test_rf, y_pred_rf, title='Random Forest Prediction')

    # XGBoost
    y_test_xgb, y_pred_xgb, xgb_model = cpp.train_rf_xgb(features, model_type='xgb')
    cpp.evaluate(y_test_xgb, y_pred_xgb)
    cpp.plot_predictions(y_test_xgb, y_pred_xgb, title='XGBoost Prediction')

    # LSTM
    X_train, X_test, y_train, y_test = cpp.prepare_lstm_data(features)
    y_test_lstm, y_pred_lstm, lstm_model = cpp.train_lstm(
        X_train, y_train, X_test, y_test, feature_len=len(features)
    )
    cpp.evaluate(y_test_lstm, y_pred_lstm)
    cpp.plot_predictions(y_test_lstm, y_pred_lstm, title='LSTM Prediction')