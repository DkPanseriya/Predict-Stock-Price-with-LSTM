# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
from tensorflow.random import set_seed

set_seed(455)
np.random.seed(455)

def load_stock_file(stock):
    stock_file = [i for i in Path(r"C:\Dk\Projects\[4.Learn] Stock Price with LSTM\stocks").iterdir() if i.name.endswith(f'{stock}.csv')][0]
    dataset = pd.read_csv(stock_file, index_col="Date", parse_dates=["Date"])
    return dataset


def train_test_plot(dataset, tstart, tend):
    dataset.loc[f"{tstart}":f"{tend}", "High"].plot(figsize=(16,4), legend=True)
    dataset.loc[f"{tend+1}":, "High"].plot(figsize=(16,4), legend=True)
    plt.legend([f"Train (Before {tend+1})", f"Test ({tend+1} and Beyond)"])
    plt.title("Mastercard stock price")
    plt.show()


def train_test_split(dataset, tstart, tend):
    train = dataset.loc[f"{tstart}":f"{tend}", "High"].values
    test = dataset.loc[f"{tend+1}":, "High"].values
    return train, test


def scale_dataset(training_dataset):
    sc = MinMaxScaler(feature_range=(0,1))
    training_set = training_dataset.reshape(-1,1)
    training_set_scaled = sc.fit_transform(training_set)
    return training_set_scaled, sc


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_idx = i + n_steps
        if end_idx > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)
    

def build_LSTM(n_steps, features):
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=125, activation="tanh", input_shape=(n_steps, features)))
    model_lstm.add(Dense(units=1))

    model_lstm.compile(optimizer="RMSprop", loss="mse")

    model_lstm.summary()
    return model_lstm


def validation(dataset, test_dataset, n_steps, features, model_lstm, sc):
    dataset_total = dataset.loc[:, "High"]
    inputs = dataset_total[len(dataset_total) - len(test_dataset) - n_steps:].values
    inputs = inputs.reshape(-1,1)

    # scaling
    inputs = sc.transform(inputs)

    # Split into samples
    X_test, y_test = split_sequence(inputs, n_steps)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)

    # prediction
    predicted_stock_price = model_lstm.predict(X_test)

    # inverse transform the values
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    plot_test_predictions(test_dataset,predicted_stock_price)
    return_rmse(test_dataset,predicted_stock_price)


def plot_test_predictions(test, predicted):
    plt.plot(test, color="grey", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("MasterCard Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("MasterCard Stock Price")
    plt.legend()
    plt.show()


def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {:.2f}.".format(rmse))


def predict_future(dataset, start_year, n_steps, model_lstm, sc, years_ahead):
    latest_data = dataset.loc[f"{start_year}":, "High"].values
    latest_data_scaled, sc = scale_dataset(latest_data)
    X_new, _ = split_sequence(latest_data_scaled, n_steps)
    X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)

    predictions = []
    for _ in range(years_ahead * 252):  # Assuming 252 trading days in a year
        pred = model_lstm.predict(X_new[-1].reshape(1, n_steps, 1))
        predictions.append(pred[0][0])
        new_input = np.append(X_new[-1][1:], pred).reshape(1, -1, 1)
        X_new = np.vstack((X_new, new_input))

    predictions = sc.inverse_transform(np.array(predictions).reshape(-1, 1))
    plot_future_predictions(dataset, start_year, predictions, years_ahead)


def plot_future_predictions(dataset, start_year, predictions, years_ahead):
    historical_data = dataset.loc[f"{start_year}":, "High"]
    future_dates = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(days=1), periods=len(predictions), freq='B')
    
    plt.figure(figsize=(16, 6))
    plt.plot(historical_data.index, historical_data.values, color="grey", label="Historical Data")
    plt.plot(future_dates, predictions, color="orange", label="Future Predictions")
    
    plt.title(f"Stock Price Prediction: Historical Data from {start_year} and {years_ahead} Years Ahead")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


def train(dataset, tstart, tend, n_steps):
    
    training_dataset, test_dataset = train_test_split(dataset, tstart, tend)
    training_set_scaled, sc = scale_dataset(training_dataset)

    features = 1
    X_train, y_train = split_sequence(training_set_scaled, n_steps)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)

    model_lstm = build_LSTM(n_steps, features)
    model_lstm.fit(X_train, y_train, epochs=50, batch_size=32)
    
    return model_lstm, test_dataset, features, sc


def main():
    stock = str(input("Enter the Stock you want to train for: ")) 
    tstart = int(input("from which year to start: "))
    tend = int(input("till which year to end: "))
    n_steps = int(input("How many steps to consider for predicting: "))

    dataset = load_stock_file(stock)
    train_test_plot(dataset, tstart, tend)

    model_lstm, test_dataset, features, sc = train(dataset, tstart, tend, n_steps)
    validation(dataset, test_dataset, n_steps, features, model_lstm, sc)

    predict_future_prices = input("Do you want to predict future prices? (Y/N): ").strip().lower()
    if predict_future_prices == 'y':
        start_year = int(input("From which year to start retraining: "))
        years_ahead = int(input("How many years ahead to predict: "))
        predict_future(dataset, start_year, n_steps, model_lstm, sc, years_ahead)


if __name__ == "__main__":
    main()