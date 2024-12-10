import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv(r"C:\Users\tanse\Downloads\TAMO Historical Data.csv")

# Display the first few rows to understand the data structure
print(df.head())

# Preprocess the data
data = df[['Date', 'Price']].copy()  # Create a copy to avoid modifying the original DataFrame

# Convert 'Date' to datetime with the correct format (day-month-year)
data.loc[:, 'Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Remove commas and convert 'Price' to a numeric type (float)
data.loc[:, 'Price'] = data['Price'].replace({',': ''}, regex=True)  # Remove commas
data.loc[:, 'Price'] = pd.to_numeric(data['Price'], errors='coerce')  # Convert to float, coerce errors to NaN

# Sort by date (just in case)
data = data.sort_values(by='Date')

# Set 'Date' as the index (optional)
data.set_index('Date', inplace=True)

# Check for missing values and handle them (e.g., dropping rows with NaN values)
data = data.dropna()

# Scale the 'Price' using MinMaxScaler (range 0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Price']])

# Split the data into training and testing sets (80% for training)
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Function to create dataset suitable for LSTM (using past 'n' days to predict the next day)
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])  # Use previous 'time_step' days to predict next day's price
        y.append(data[i, 0])  # Predict the next day's price
    return np.array(X), np.array(y)

# Prepare datasets for LSTM input
X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Reshape data for LSTM input: [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Save processed data to use in training the model
np.savez('processed_data.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scaler=scaler)
