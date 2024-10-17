
# Stock Price Prediction with LSTM

This repository focuses on predicting stock prices using historical data and machine learning models, specifically Long Short-Term Memory (LSTM) neural networks. It includes scripts for data preprocessing, model training, prediction, and visualization.

## Project Overview

The project trains a machine learning model using historical stock data to predict future stock prices. It uses LSTM and TensorFlow to model time-series data for stock prices (Open, High, Low, Close, Volume). The project also supports scaling, splitting, and visualizing the data, and offers functionality to predict future stock prices based on the trained model.

The historical stock data can be downloaded using the notebook sourced from [this Kaggle notebook](https://www.kaggle.com/code/jacksoncrow/download-nasdaq-historical-data), which fetches stock data from the NASDAQ.

### Features
- Stock price prediction using LSTM.
- Training and testing with custom date ranges.
- Data visualization for both training and predicted data.
- Predict future stock prices for the desired number of years ahead.
- GPU support for faster training and prediction.
- Utility for downloading historical stock data from NASDAQ.

## Requirements

To run this project, ensure that the following dependencies are installed:

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- yfinance (for downloading historical data)

Install the necessary libraries via:

```bash
pip install -r requirements.txt
```

### Required Libraries

```bash
tensorflow
numpy
pandas
matplotlib
scikit-learn
yfinance
```

## How to Use

1. **Download Stock Data:**
   You can download NASDAQ stock data by running the `download_nasdaq_historical_data.ipynb` notebook. This script will fetch historical data for all NASDAQ-traded symbols and separate stocks from ETFs.

2. **Run the Prediction Script:**
   You can run the main script (`main.py`) to train the LSTM model, test it, and predict stock prices for future dates. Ensure the stock data is stored in the appropriate directory before running the script.

   ```bash
   python main.py
   ```

3. **Training the Model:**
   The model uses historical data from `.csv` files to train the LSTM model. The script prompts you for the stock symbol, training period, and the number of steps (lag days) to predict stock prices.

4. **Predicting Future Prices:**
   After training, you can choose to predict future stock prices for a specified number of years.

## Configurations

- **Stock Symbol**: Specify which stock symbol to use (e.g., 'AAPL').
- **Training Period**: Define the start and end year for training the model.
- **Future Prediction**: Option to predict stock prices for a certain number of years ahead after model training.

## Folder Structure

- `stocks/`: Contains the stock data used for training and prediction.
- `etfs/`: Contains data for exchange-traded funds (ETFs), which can be excluded from stock prediction tasks.
- `scripts/`: Contains all Python scripts for training, prediction, and visualization.
- `notebooks/`: Contains Jupyter notebooks for downloading and preprocessing data.

## Example Usage

```bash
# Run the main script and follow the prompts
python main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

Feel free to adjust the content according to your project specifics.
