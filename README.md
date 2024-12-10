# Stock Price Prediction with LSTM - README

This project demonstrates how to use a Long Short-Term Memory (LSTM) model to predict future stock prices. The project involves three main steps: preprocessing the data, training the LSTM model, and making predictions using new stock data. Below is a detailed explanation of the three parts of the project.

## Prerequisites

Before running any of the scripts, ensure that the following libraries are installed:

- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`
- `tensorflow`
- `pickle`

You can install the necessary libraries using the following command:

pip install numpy pandas matplotlib scikit-learn tensorflow


## File Overview

### 1. **Data Preprocessing and Model Training (First and Second Files)**
   **File Name:** `train_model.py` (or whatever you prefer, both the first and second scripts share similar logic).

   **Purpose:** This script processes historical stock price data, scales it, and trains an LSTM model to predict the future price of the stock. It then saves the trained model and the scaler for later use in prediction.

#### **Steps:**
1. **Load Data:**
   - Loads historical stock data from a CSV file.
   - The CSV file should contain at least two columns: 'Date' and 'Price'.
   - The 'Price' column may have commas (e.g., "1,234.56"), which are removed.

2. **Data Preprocessing:**
   - Converts the 'Date' column to `datetime` format.
   - Removes any commas from the 'Price' column and converts it into a numerical format.
   - Sorts the data by 'Date' and sets it as the index.
   - Drops any rows with missing values.

3. **Feature Scaling:**
   - Uses a MinMaxScaler to scale the 'Price' values between 0 and 1, which is important for neural networks like LSTMs.

4. **Create Dataset for LSTM:**
   - Uses a sliding window approach (last 60 days) to create input sequences for the LSTM model to predict the next day's price.

5. **Train-Test Split:**
   - Splits the data into training (80%) and testing (20%) sets.

6. **Build and Train LSTM Model:**
   - Defines a Sequential model with two LSTM layers, Dropout for regularization, and a Dense output layer.
   - The model is trained using the training data (`X_train`, `y_train`) for 20 epochs.

7. **Save Model and Scaler:**
   - Saves the trained model as `stock_prediction_model.h5` and the scaler as `scaler.pkl`.


### 2. **Model Training and Saving (Second File)**
   **File Name:** `train_model_v2.py` (this file serves as an alternative version with the same basic logic for training).

   **Purpose:** This script performs the same functions as the first one. It is designed to ensure that the model and scaler are correctly trained and saved.

#### **Steps:**
- Similar to the first file, this script:
  - Loads the stock price data.
  - Preprocesses the data (removes commas, converts to numeric, and scales the data).
  - Splits the data into training and testing sets.
  - Builds and trains the LSTM model.
  - Saves the trained model and scaler for future predictions.


### 3. **Prediction on New Data (Third File)**
   **File Name:** `predict_stock_price.py`

   **Purpose:** This script loads the trained model and scaler, processes new stock data, and makes a prediction about the next day's stock price.

#### **Steps:**
1. **Load Trained Model and Scaler:**
   - Loads the trained LSTM model (`stock_prediction_model.h5`) and scaler (`scaler.pkl`) from disk.

2. **Load New Data for Prediction:**
   - Loads new stock price data (CSV file) that contains recent stock prices for making predictions.

3. **Preprocess New Data:**
   - Similar to the training script, the 'Price' column is cleaned (removes commas, converts to numeric).
   - The new data is scaled using the saved scaler to ensure consistency with the model input.

4. **Prepare Data for Prediction:**
   - The script uses the most recent 60 days of stock price data to create a sequence (just as in training).
   - If the new data contains less than 60 days, the script will print an error message indicating insufficient data.

5. **Make Prediction:**
   - The LSTM model predicts the next day’s stock price based on the last 60 days of data.
   - The prediction is then inverse-transformed using the scaler to get the predicted price in the original scale.

6. **Output the Predicted Price:**
   - The predicted price is printed to the console.


## How to Use

### Step 1: Train the Model

Run either of the following scripts to train the LSTM model on historical stock price data:

python train_model.py

or

python train_model_v2.py


This will:
- Load the stock data from `TAMO Historical Data.csv` (make sure this CSV file is properly formatted).
- Preprocess the data (clean and scale it).
- Train the LSTM model.
- Save the model as `stock_prediction_model.h5` and the scaler as `scaler.pkl`.

### Step 2: Make Predictions

After training the model, you can use the following script to make predictions on new stock data:

python predict_stock_price.py


This script will:
- Load the trained model and scaler from the disk.
- Load new stock data from `TAMO Historical Data (1).csv`.
- Preprocess and scale the new data.
- Predict the next stock price based on the last 60 days of data.
- Output the predicted next price.

### File Structure

Ensure your directory has the following files:

/project-directory
    ├── train_model.py
    ├── train_model_v2.py
    ├── predict_stock_price.py
    ├── TAMO Historical Data.csv  # Historical data used for training
    ├── TAMO Historical Data (1).csv  # New data for predictions
    ├── stock_prediction_model.h5  # Trained model (after running training script)
    └── scaler.pkl  # Scaler object (after running training script)


### Example CSV Format

Your CSV files should have at least the following columns:

Date,Price
01-01-2019,1200.50
02-01-2019,1215.75


Where `Date` is in the format `dd-mm-yyyy`, and `Price` represents the stock price on that date.


## Troubleshooting

- **Insufficient Data for Prediction:** If the new data contains fewer than 60 days of stock prices, the script will not be able to generate predictions. Make sure you provide a CSV with enough data for at least 60 days.
- **Model Not Found:** Ensure that the model file (`stock_prediction_model.h5`) and scaler file (`scaler.pkl`) are present in the same directory as the prediction script. These files are generated after training.


## Conclusion

This project demonstrates the full workflow for training an LSTM model on stock price data, saving the model and scaler, and using them to make predictions on new stock data. By following these steps, you can easily train and predict future stock prices using a deep learning model.
