# Stock Price Prediction using LSTM and GRU

This project demonstrates how to predict stock prices using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) neural networks. The model is built using TensorFlow and Keras, and trained on historical stock market data. The project also includes features for visualizing stock price trends, scaling data, and forecasting future prices.

## Project Structure

- **Data Loading**: Load historical stock data from CSV files.
- **Data Visualization**: Visualize the stock's open, high, low, close prices, and volume.
- **Data Preprocessing**: Scale the stock data and prepare it for LSTM input.
- **Model Building**: Build, train, and evaluate the LSTM model for stock price prediction.
- **Validation**: Test the model on unseen stock data and evaluate the accuracy.
- **Prediction**: Predict future stock prices for a given number of years ahead.

## Technologies Used

- **Libraries**:
  - `numpy`: For numerical operations.
  - `pandas`: For data manipulation.
  - `matplotlib`: For data visualization.
  - `scikit-learn`: For data scaling and error calculation.
  - `tensorflow`: For building and training the LSTM neural networks.

## Features

- **GPU Compatibility**: The code automatically checks for available GPUs and configures TensorFlow to use them for faster training.
- **Stock Prediction**: Train on stock data and predict future prices based on historical trends.
- **Visualization**: Plot both the historical stock prices and the model's predictions for better insights.

## How to Run

1. **Install Dependencies**:  
   You need Python 3.x and the required libraries. Install the dependencies by running:

   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow
   ```

2. **Prepare Stock Data**:  
   Ensure your stock data is available as CSV files in the `stocks/` directory with columns: `Date, Open, High, Low, Close, Volume`.

3. **Run the Script**:  
   You can run the script by executing:

   ```bash
   python stock_prediction.py
   ```

4. **Input Parameters**:
   - Stock symbol (file name without `.csv`).
   - Start and end year for training.
   - Number of time steps (sequence length) for prediction.
   - Optionally, predict future prices after model training.

## Key Functions

- `load_stock_file(stock)`: Loads the CSV file for the specified stock symbol.
- `train_test_plot(dataset, tstart, tend)`: Visualizes the training and test stock data.
- `train_test_split(dataset, tstart, tend)`: Splits the dataset into training and test sets.
- `scale_dataset(training_dataset)`: Scales the dataset using MinMaxScaler.
- `build_LSTM(n_steps, features)`: Builds and compiles the LSTM model.
- `validation()`: Evaluates the model and plots test predictions.
- `predict_future()`: Predicts stock prices for a specified number of years ahead.

## Example Usage

```python
# Example stock symbol and parameters
Enter the Stock you want to train for: AAPL
from which year to start: 2015
till which year to end: 2020
How many steps to consider for predicting: 60
```

After running, the script will:
- Load and visualize historical stock data.
- Train the LSTM model on the data.
- Test the model on unseen data and plot the predictions.
- Optionally, predict stock prices for future years.

## Model Architecture

The LSTM model is built with the following layers:
- LSTM layers with 125 and 64 units.
- Dense output layer for predicting `Open`, `High`, `Low`, `Close`, and `Volume`.

The model is trained using the Adam optimizer and mean squared error (MSE) loss function.

## Requirements

- Python 3.x
- TensorFlow >= 2.x
- pandas
- numpy
- scikit-learn
- matplotlib

## Results

The project plots both the real and predicted stock prices. Additionally, it provides root mean squared error (RMSE) for each stock price component (`Open`, `High`, `Low`, `Close`, and `Volume`), allowing you to assess model accuracy.

## License

This project is open-source and available under the [MIT License](LICENSE).

```

This `README.md` provides an overview of your project, including setup instructions, a summary of the functionality, and example usage. Let me know if you'd like any specific details added!
