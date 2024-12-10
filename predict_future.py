import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Step 1: Load the model and scaler for prediction
model = tf.keras.models.load_model('stock_prediction_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Step 2: Load the most recent data for prediction (Replace with your file path)
new_data = pd.read_csv(r"C:\Users\tanse\Downloads\TAMO Historical Data (1).csv")  # Replace with your actual CSV file

# Step 3: Clean the 'Price' column
new_data['Price'] = new_data['Price'].replace({',': ''}, regex=True)
new_data['Price'] = pd.to_numeric(new_data['Price'], errors='coerce')
new_data = new_data.dropna(subset=['Price'])

# Step 4: Scale the new data using the trained scaler
scaled_new_data = scaler.transform(new_data['Price'].values.reshape(-1, 1))

# Debug: Check the shape of scaled_new_data
print(f"Shape of scaled_new_data: {scaled_new_data.shape}")

# Step 5: Prepare the data in the same way as training (use the last 60 days)
time_steps = 60
X_new = []

# Ensure there is enough data for the last 60 days
if len(scaled_new_data) < time_steps:
    print(f"Insufficient data for 60 time steps. Need at least 60 data points, but got {len(scaled_new_data)}.")
else:
    # Collect 60-day sequences, starting from (len(scaled_new_data) - time_steps) to the end
    for i in range(len(scaled_new_data) - time_steps, len(scaled_new_data)):
        # Extract the last 60 days of data (from i - 60 to i)
        current_sequence = scaled_new_data[i - time_steps:i, 0]  # Extract values as 1D array
        if current_sequence.shape[0] == time_steps:
            # Ensure it's exactly 60 days worth of data
            current_sequence = current_sequence.reshape(-1, 1)  # Reshape to match model input
            print(f"Shape of current_sequence at index {i}: {current_sequence.shape}")  # Debug each sequence's shape
            X_new.append(current_sequence)

# Debug: Check the length of X_new and the first item in X_new
print(f"Length of X_new: {len(X_new)}")
if X_new:
    print(f"Shape of the first item in X_new: {np.array(X_new[0]).shape}")
else:
    print("X_new is empty!")

# Convert the list of 60-day sequences into a numpy array
try:
    X_new = np.array(X_new)
except Exception as e:
    print(f"Error when converting to numpy array: {e}")
    X_new = []

# Debug: Check the shape of X_new after conversion
if len(X_new) > 0:
    print(f"Shape of X_new after conversion: {X_new.shape}")
else:
    print("X_new is empty!")

# Ensure X_new is reshaped as (samples, time_steps, 1) for LSTM input
if len(X_new) > 0 and X_new.ndim == 3:
    X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)
else:
    print("Error: X_new is not in the correct shape.")

# Step 6: Predict the next price using the model
if len(X_new) > 0:
    predicted_price = model.predict(X_new)
    predicted_price = scaler.inverse_transform(predicted_price)

    # Step 7: Output the predicted price
    print(f"Predicted next price: {predicted_price[0][0]:.2f}")
else:
    print("Failed to process data for prediction.")
