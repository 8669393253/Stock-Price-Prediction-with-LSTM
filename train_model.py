import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pickle

# Step 1: Load your data (Replace with your actual file path)
data = pd.read_csv(r"C:\Users\tanse\Downloads\TAMO Historical Data.csv")  # Data from 2019 to 2024

# Step 2: Clean the 'Price' column if needed (remove commas, convert to float)
data['Price'] = data['Price'].replace({',': ''}, regex=True)  # Remove commas
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')  # Convert to numeric, invalid data will be NaN

# Drop rows with missing values
data = data.dropna(subset=['Price'])

# Step 3: Scale the data (use MinMaxScaler for consistency in input-output scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))

# Step 4: Prepare the data for LSTM with a sliding window of 60 days
time_steps = 60
X = []
y = []

# Use a sliding window to gather the last 60 days of data to predict the next day
for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i - time_steps:i, 0])  # Last 60 days
    y.append(scaled_data[i, 0])  # The next day's price

X = np.array(X)
y = np.array(y)

# Reshape X to 3D (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 5: Train/Test split based on your needs
# In this case, let's split the data at the last 6 months or the last 20% for testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 6: Build the LSTM model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(tf.keras.layers.Dropout(0.2))  # Dropout to prevent overfitting
model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 7: Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the model and scaler
model.save('stock_prediction_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
