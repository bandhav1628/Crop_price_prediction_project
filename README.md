# Crop Price Prediction Project

Predict crop market prices using AI/ML models (ARIMA, Prophet, Random Forest, LSTM) on historical price and weather data.

## Data Structure

- `data/market_prices.csv`: Columns `[date, location, price]`
- `data/weather_data.csv`: Columns `[date, location, weather_temp, weather_rain]`

## Models Used

- ARIMA (time series)
- Prophet (seasonality, holidays)
- Random Forest (tabular regression)
- LSTM (deep learning time series)

## How to Run

1. Place data files in `data/` directory.
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run:
   ```
   python main.py
   ```

## Output

- Prints metrics (MAE, RMSE) for each model.
- Plots actual vs. predicted prices.

## Extending

- Add more features (yield, economic indicators).
- Deploy as API (Flask/Django).
- Build dashboard (Streamlit/Dash).

## References

- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [TensorFlow LSTM Guide](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
