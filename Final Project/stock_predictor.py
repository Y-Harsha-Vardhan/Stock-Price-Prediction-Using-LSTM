# Importing necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

# Suppressing warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Task 1: Data Collection ---
## Getting user input for stock data
print("--- Stock Price Prediction using LSTM ---")
stock_name = input("Enter the stock ticker (e.g., 'AAPL', 'GOOGL'): ")
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")
# For simplicity, we'll use a daily timeframe ('1d') as it's most common for this type of analysis.
# timeframe = input("Enter the timeframe (e.g., '1d', '1wk', '1mo'): ")
timeframe = '1d'

# Fetching stock data using yfinance
try:
    data = yf.download(stock_name, start=start_date, end=end_date, interval=timeframe)
    if data.empty:
        raise ValueError("No data found for the given stock ticker and date range.")
    print(f"\nSuccessfully fetched data for {stock_name}.")
    print(data.head())
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# --- Task 2: Data Visualization & Technical Indicators ---
## Calculating technical indicators: MACD and RSI
# 1. MACD (Moving Average Convergence Divergence)
exp1 = data['Adj Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Adj Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = exp1 - exp2
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# 2. RSI (Relative Strength Index)
delta = data['Adj Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Dropping the rows with NaN values created by indicators
data.dropna(inplace=True)

## Visualizing the data
print("\nVisualizing historical data and technical indicators...")

# Plot 1: Adjusted Closing Price
plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)
plt.plot(data['Adj Close'], label='Adjusted Closing Price', color='blue')
plt.title(f'{stock_name} Adjusted Closing Price')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

# Plot 2: MACD
plt.subplot(3, 1, 2)
plt.plot(data['MACD'], label='MACD', color='red')
plt.plot(data['Signal_Line'], label='Signal Line', color='green')
plt.title('MACD (Moving Average Convergence Divergence)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Plot 3: RSI
plt.subplot(3, 1, 3)
plt.plot(data['RSI'], label='RSI', color='purple')
plt.title('RSI (Relative Strength Index)')
plt.axhline(70, linestyle='--', color='orange', label='Overbought (70)')
plt.axhline(30, linestyle='--', color='brown', label='Oversold (30)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# --- Task 3: Data Preprocessing ---
## Preparing data for LSTM model
# Selecting features: Adjusted Close, MACD, and RSI
features = ['Adj Close', 'MACD', 'RSI']
dataset = data[features].values

# Normalizing the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Creating sequences
sequence_length = 60 # Using past 60 days to predict the next
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, :])
    y.append(scaled_data[i, 0]) # Target is the 'Adj Close' price

X, y = np.array(X), np.array(y)

# Splitting the data into training and testing sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\nData preprocessed and split into training and testing sets.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# --- Task 4: Build and Train the LSTM Model ---
print("\nBuilding and training the LSTM model...")

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compiling and training the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=25)


# --- Task 5: Prediction, Visualization, and Evaluation ---
print("\nMaking predictions and evaluating the model...")

## Predicting on the test set
predictions = model.predict(X_test)

## Inversing the transform predictions to get actual price values
# We need to create a dummy array with the same number of features to inverse transform
dummy_array_predictions = np.zeros((len(predictions), len(features)))
dummy_array_predictions[:, 0] = predictions.flatten()
predicted_prices = scaler.inverse_transform(dummy_array_predictions)[:, 0]

# Doing the same for the actual values (y_test)
dummy_array_actual = np.zeros((len(y_test), len(features)))
dummy_array_actual[:, 0] = y_test.flatten()
actual_prices = scaler.inverse_transform(dummy_array_actual)[:, 0]

## Visualizing predictions vs. actual values
plt.figure(figsize=(15, 7))
plt.plot(actual_prices, color='red', label='Actual Stock Price')
plt.plot(predicted_prices, color='green', label='Predicted Stock Price')
plt.title(f'{stock_name} Stock Price Prediction')
plt.xlabel('Time (days)')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

## Calculating R² score
r2 = r2_score(actual_prices, predicted_prices)
print(f"\nModel Performance Evaluation:")
print(f"R² Score: {r2:.4f}")

## Forecasting the next 15 units (days)
print("\nForecasting the next 15 days...")

# Getting the last sequence from the original scaled data
last_sequence = scaled_data[-sequence_length:]
forecast = []

for _ in range(15):
    # Reshaping the sequence for prediction
    current_sequence = np.reshape(last_sequence, (1, sequence_length, len(features)))
    
    # Predicting the next value
    predicted_scaled_price = model.predict(current_sequence)[0, 0]
    
    # We need to generate the other features (MACD, RSI) for the new predicted point.
    # This is complex. For simplicity, we'll append a dummy row and update only the price.
    # A more advanced approach would re-calculate indicators based on the new predicted price.
    new_row = np.zeros((1, len(features)))
    new_row[0, 0] = predicted_scaled_price
    # We can use the last known values for other features as a simple approximation
    new_row[0, 1:] = last_sequence[-1, 1:]
    
    # Appending the predicted value to the forecast list
    forecast.append(predicted_scaled_price)
    
    # Updating the sequence for the next prediction
    last_sequence = np.append(last_sequence[1:], new_row, axis=0)

# Inverse transforming the forecast
dummy_array_forecast = np.zeros((len(forecast), len(features)))
dummy_array_forecast[:, 0] = forecast
forecasted_prices = scaler.inverse_transform(dummy_array_forecast)[:, 0]

print("Forecasted Prices for the next 15 days:")
print(forecasted_prices)

## Visualizing the forecast
# Getting the dates for the forecast
last_date = data.index[-1]
forecast_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, 16)])

plt.figure(figsize=(15, 7))
plt.plot(data.index[train_size+sequence_length:], actual_prices, color='red', label='Historical Actual Price')
plt.plot(forecast_dates, forecasted_prices, color='blue', linestyle='--', marker='o', label='Forecasted Price (Next 15 Days)')
plt.title(f'{stock_name} Price Forecast')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Project Completed ---")