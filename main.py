# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Step 1: Download historical stock data
ticker = "AAPL"  # Example: Apple Inc.
data = yf.download(ticker, start="2000-01-01", end="2024-10-01")
data.to_csv(f"{ticker}_stock_data.csv")

# Step 2: Simulate news sentiment data
def simulate_sentiment(data):
    # Generate random sentiment scores between -1 and 1
    np.random.seed(42)
    sentiment = np.random.uniform(low=-1, high=1, size=len(data))
    return sentiment

# Add simulated sentiment to the stock data
data['Sentiment'] = simulate_sentiment(data)

# Step 3: Feature engineering
features = data[['Close', 'Sentiment']]

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Split data into training and testing sets
train_size = int(len(scaled_features) * 0.8)
train_data, test_data = scaled_features[:train_size], scaled_features[train_size:]

# Create sequences with sentiment
def create_sequences_with_sentiment(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, :])  # Include both 'Close' and 'Sentiment'
        y.append(data[i, 0])  # Predict 'Close' price
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_sequences_with_sentiment(train_data, time_step)
X_test, y_test = create_sequences_with_sentiment(test_data, time_step)

# Step 4: Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Step 5: Evaluate the model
predictions = model.predict(X_test)

# Inverse transform the predictions
predictions = scaler.inverse_transform(np.concatenate((predictions, X_test[:, -1, 1].reshape(-1, 1)), axis=1))[:, 0]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(test_data[time_step:])[:, 0], predictions))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 6: Predict future prices with sentiment
future_steps = 30
last_sequence = X_test[-1]
future_predictions = []

for _ in range(future_steps):
    next_pred = model.predict(last_sequence.reshape(1, time_step, 2))
    future_predictions.append(next_pred[0, 0])
    last_sequence = np.append(last_sequence[1:], [[next_pred[0, 0], 0]], axis=0)  # Assume neutral sentiment for future

# Inverse transform future predictions
future_predictions = scaler.inverse_transform(np.concatenate((np.array(future_predictions).reshape(-1, 1), np.zeros((future_steps, 1))), axis=1))[:, 0]

print("Future Predictions:")
print(future_predictions)

# Step 7: Visualize the results
plt.figure(figsize=(14, 5))
plt.plot(data.index[-len(y_test):], scaler.inverse_transform(test_data[time_step:])[:, 0], label='Actual Prices')
plt.plot(data.index[-len(y_test):], predictions, label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.legend()
plt.show()