import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from datetime import date, datetime, timedelta
import os
from newsapi.newsapi_client import NewsApiClient
import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
import hvplot.pandas
from joblib import dump, load

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

from sklearn.preprocessing import MinMaxScaler


def get_api():
    '''
    NOTE: Need to make sure this works with env variables on AWS.
    '''
    # Set News API Key
    newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

    # Set Alpaca API key and secret
    alpaca_api_key = os.getenv("ALPACA_API_KEY_ID")
    alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")

    api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, api_version='v2')
    
    return api



# Split data by time segments of size `window` days
def window_data(stock_df, window, feature_col_num, target_col_num):
    X = []
    y = []
    
    for i in range(len(stock_df) - window):
        X_comp = stock_df.iloc[i:(i+window), feature_col_num]
        y_comp = stock_df.iloc[(i+window), target_col_num]
        X.append(X_comp)
        y.append(y_comp)
    return np.array(X), np.array(y).reshape(-1, 1)



# Construct model
def create_model_and_dataset(ticker, fit_window=2):
    '''
    Get stock data, creates model, trains model and returns 
    model for prediction.
    
    Since we need stock data in `predicted_porfolio_metrics`,
    and we're pulling the data here, we just return the stock
    data along with the model.
    
    `ticker` is stock ticker, such as 'GOOGL' or 'AMZN'.
    '''    
    # Set timeframe to '1D'
    timeframe = '1D'

    # Get current date and the date from one month ago
    current_date = date.today()
    past_date = date.today() - timedelta(weeks=4)

    # Get stock data
    api = get_api()
    stock_df = api.get_barset(
        ticker,
        timeframe,
        limit=1000,
        start=current_date,
        end=past_date,
        after=None,
        until=None,
    ).df
    stock_df = stock_df.droplevel(0, axis=1)
    stock_df.drop(columns=['open', 'high', 'low', 'volume'], inplace=True)
    stock_df.index = stock_df.index.date
    
    
    # Create feature & target
    # fit_window is how many priors we use
    # in each training instance.
    X, y = window_data(stock_df, fit_window, 0, 0)

    # Train/test split
    train_size = 0.80
    split = int(len(X) * train_size)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]
    
    # Scale data
    scaler = MinMaxScaler()

    scaler.fit(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    scaler.fit(y)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)

    # Reshape the features for the model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    

    # Instantiate model
    model = Sequential()

    # Fitting parameters
    num_inputs = fit_window
    dropout_fraction = 0.2

    # Layer 1
    model.add(LSTM(
        units=num_inputs,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
    ))
    model.add(Dropout(dropout_fraction))
    # Layer 2
    model.add(LSTM(
        units=num_inputs,
        return_sequences=True
    ))
    model.add(Dropout(dropout_fraction))
    # Layer 3
    model.add(LSTM(
        units=num_inputs
    ))
    model.add(Dropout(dropout_fraction))
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Fit model
    model.fit(
        X_train, 
        y_train, 
        batch_size=2, 
        epochs=5, 
        shuffle=False,
        verbose=0,
        validation_data=(X_test, y_test)
    )
    
    return model, stock_df



def export_model(ticker, model, current_date=datetime.today()):
    '''
    Model should be exported iff it is up-to-date, i.e.
    has been fitted to the most recent data. Therefore we
    pass the date in to know whether a saved model is current.
    '''
    model_json = model.to_json()
    model_path = Path(f"./Models/{ticker}_model.json")
    model.save(model_path)



def get_model(ticker):
    # Path should take a regular expression to id the model,
    # because the file name will have a date (see `export_model`)
    # and we'll need to parse it to know whether we need
    # to update the model.
    model_path = Path(f"./Models/{ticker}_model.json")
    model = load_model(model_path)
    return model



def prep_data_for_fitting(df, fit_window=2):
    '''
    Create data using `window_data`.
    Scale data.
    Return X, y for fitting model.
    '''
    X, y = window_data(df, fit_window, 0, 0)
    
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    scaler.fit(y)
    y = scaler.transform(y)
    
    return X, y



def prep_data_for_pred(data, stock_data, fit_window=2):
    '''
    Prep data for prediction.
    
    `stock_data` is need to fit the scaler to ALL of the data.
    '''
    X = np.array(data)
    
    scaler = MinMaxScaler()
    scaler.fit(stock_data)
    X = scaler.transform(X)
    
    # Reshape for model
    # Not sure how to set this programmatically (e.g. using
    # X.shape[0], X.shape[1], etc.) but this works.
    X = X.reshape((1, fit_window, 1))
    
    return X



def update_model(model):
    '''
    Fits model with most current data.
    '''
    # TODO
    pass



def rescale_pred(pred, stock_data):
    '''
    Inversely scale data.
    '''
    scaler = MinMaxScaler()
    scaler.fit(stock_data)
    pred = scaler.inverse_transform(pred)
    return pred



def predicted_portfolio_metrics(model, all_data, window=30, fit_window=2):
    '''
    Uses `model` to predict returns `window` days in the future.
    
    Expect `window` from user, but default to 30. This is the holding
    window, i.e. how far out we want to predict.
    '''
    
    # DF where we'll append predictions as we generate them. Copying 
    # most recent `training_window` days from all_data so that we 
    # have a basis for predictions.
    df = all_data.iloc[-fit_window:].copy()

    for _ in range(window):        
        # Scale data and reshape for model
        #
        # Still using fit_window here because we only want
        # most recent fit_window days for making prediction
        data = prep_data_for_pred(df.iloc[-fit_window:], all_data, fit_window)
        
        # Get prediction
        pred = model.predict(data)
        
        # Inverse transform and reshape predicted price
        pred = rescale_pred(pred, all_data)
        pred = pred.reshape(-1,)
        
        # Get date for adding new row to index
        pred_date = df.iloc[-1].name + timedelta(days=1)
        
        # Add predictions to df
        df = df.append(pd.DataFrame({'close': pred}, index=[pred_date]))
        
        # Fit model with added data (predictions)
        X, y = prep_data_for_fitting(df, fit_window)
        model.fit(X, y, epochs=10, shuffle=False, batch_size=1, verbose=0)
        
    # After for loop, last row in df should be the prediction 
    # that we want.
    pred_to_return = df.iloc[-1]
    
    # TODO
    # Calculate pred_return, sharpe ratio, date of prediction (just to 
    # be transparent for user)
    df['return'] = df['close'].pct_change()
    pred_return = df.iloc[-1]['return'] * 100
    
    sharpe_ratio = (df['return'].mean() * 252) / (df['return'].std() * np.sqrt(252))
    
    prediction_date = df.iloc[-1].name
    
    return pred_return, sharpe_ratio, prediction_date



def get_portfolio_predictions(tickers):
    '''
    *** Entry point into module.
    '''
    
    predicted_values = dict.fromkeys(tickers)
    
    for ticker in tickers:
        # Get model if exists, otherwise create one
        
        model, data = create_model_and_dataset(ticker)
        
        pred_return, sharpe_ratio, prediction_date = predicted_portfolio_metrics(model, data)
        
        predicted_values[ticker] = [pred_return, sharpe_ratio, prediction_date]
        
    return predicted_values



#### TESTS

if __name__ == '__main__':
    portfolio = ['AMZN']
    print(get_portfolio_predictions(portfolio))
