import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from datetime import date, datetime, timedelta
import os
import alpaca_trade_api as tradeapi

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

from sklearn.preprocessing import MinMaxScaler

# Set fit window globally so it doesn't have to be passed
# around so much.
fit_window = 2


def get_api():
    '''
    NOTE: Need to make sure this works with env variables on AWS.
    '''
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



def get_data(ticker, days_back=1000, from_date=None):
    '''
    Get data from Alpaca for training model.
    
    `ticker` is stock ticker string, such as 'GOOGL' or 'AMZN'.
    
    #NOTE: really need to get `limit` for api.get_barset and not
    just the date.
    
    `from_date` was going to be the farthest date back we
    want to get data for, but instead for now I'm using `days_back`.
    '''
    # Set timeframe to '1D'
    timeframe = '1D'

    # Get current date and the date from one month ago
    current_date = date.today()
    past_date = from_date or date.today() - timedelta(weeks=4)
    
    # limit in api.get_barset must be less than 1000, but if 
    # we get it from user we need to add 1 to it.
    if days_back < 1000:
        days_back += 1

    # Get stock data
    api = get_api()
    stock_df = api.get_barset(
        ticker,
        timeframe,
        # Adding a day because it starts at today
        limit=days_back,
        #start=current_date,
        #end=past_date,
        after=None,
        until=None,
    ).df
    stock_df = stock_df.droplevel(0, axis=1)
    stock_df.drop(columns=['open', 'high', 'low', 'volume'], inplace=True)
    stock_df.index = stock_df.index.date
    
    return stock_df



# Construct model
def create_model_and_dataset(ticker, fit_window=2):
    '''
    Get stock data, creates model, trains model and returns 
    model for prediction.
    
    Since we need stock data in `predicted_porfolio_metrics`,
    and we're pulling the data here, we just return the stock
    data along with the model.
    '''
    # Get stock data
    stock_df = get_data(ticker)
    
    # Create feature & target
    # `fit_window` is how many priors we use
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
    
    # Create model
    model = build_model_keras(X_train, X_test, y_train, y_test, fit_window)
    
    return model, stock_df



def build_model_keras(X_train, X_test, y_train, y_test, fit_window):
    '''
    Pulled this out of `create_model_and_dataset` in case we
    want to try using sklearn.MLPRegressor, in which case
    we'll have a separate function for building that.
    '''
    # Reshape the features for the model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Instantiate model
    model = Sequential()

    # Model parameters
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
        verbose=1,
#         validation_data=(X_test, y_test)
    )
    
    return model



def export_model(ticker, model, current_date=datetime.today()):
    '''
    Model should be exported iff it is up-to-date, i.e.
    has been fitted to the most recent data. Therefore we
    pass the date in to know whether a saved model is current.
    
    Note: currently not implemented to do anything useful for
    our purposes.
    
    #TODO Figure out how to save the date for a model so that
    we know what data to train it on when we update it.
    
    #UPDATE Collin suggested saving a dataset csv along with the
    model and getting the last date from that.
    '''
    # Save model
    model_json = model.to_json()
    model_path = Path(f"./Models/{ticker}_model.json")
    model.save(model_path)
    
def export_dataset(ticker, dataset):
    '''
    Save dataset.
    '''
    data_path = Path(f"./Data/{ticker}.csv")
    dataset.to_csv(data_path)



def get_model_and_data(ticker):
    '''
    Basically just a wrapper around `load_model`. Until we
    figure out how to export a model with a date, this
    isn't useful for much.

    `Path` could take a regular expression to id the model,
    if the filename contained a date (see `export_model`)
    and we could parse it to know whether we need
    to update the model.
    '''
    
    # Get data so we know what dates to update model with.
    stock_path = Path(f"./Data/{ticker}.csv")
    try:
        stock_data = pd.read_csv(
            stock_path, 
            index_col=0, 
            parse_dates=True, 
            infer_datetime_format=True) # DataFrame

        # stock_data.index = stock_data.index.date
    except FileNotFoundError:
        stock_data = get_data(ticker)
        # Export data, since none has been saved for this stock yet
        export_dataset(ticker, stock_data)

    # Retrieve model
    model_path = Path(f"./Models/{ticker}_model.json")
    try:
        model = load_model(model_path)
    except:
        # If we don't have a model, create one
        model, stock_data = create_model_and_dataset(ticker)
        # Export newly created model
        export_model(ticker, model)
        # Ok to return here, since we just created a dataset
        # with latest data (no need to do the check/update below)
        return model, stock_data

    most_recent_train_date = stock_data.iloc[-1].name.date()

    today_date = datetime.today().date()

    if most_recent_train_date != today_date:
        # Calculate how many days we need to go back
        days_back = (today_date - most_recent_train_date).days

        # Get data the model hasn't seen
        latest_data = get_data(ticker, days_back)

        # Since there's new data not saved, add to df
        stock_data = stock_data.append(latest_data)

        # Export df with latest data
        export_dataset(ticker, stock_data)

        # Update model
        model = update_model(model, latest_data)

        # Export model
        export_model(ticker, model)

    # Caller is expecting model and data back
    return model, stock_data



def fit_model(model, stock_df):
    '''
    Given an existing model and data, fit model to data.
    '''
    # Create feature & target
    # `fit_window` is how many priors we use
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
    
    # Fit model
    model.fit(
        X_train, 
        y_train, 
        batch_size=2, 
        epochs=5, 
        shuffle=False,
        verbose=1,
        validation_data=(X_test, y_test)
    )
    
    return model


def update_model(model, latest_data):
    '''
    Fits model with most current data.
    '''
    model = fit_model(model, latest_data)
    
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
    
    # Reshape for model
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    scaler.fit(y)
    y = scaler.transform(y)
    
    return X, y



def prep_data_for_pred(data, stock_data, fit_window=2):
    '''
    Prep data for prediction.
    
    `stock_data` is needed to fit the scaler to ALL of the data.
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



def rescale_pred(pred, stock_data):
    '''
    Inversely scale data (viz. predicted prices).
    '''
    scaler = MinMaxScaler()
    scaler.fit(stock_data)
    pred = scaler.inverse_transform(pred)
    return pred



def predicted_portfolio_metrics(model, stock_data, window=30, fit_window=2):
    '''
    Uses `model` to predict returns `window` days in the future.
    
    Expect `window` from user, but default to 30. This is the holding
    window, i.e. how far out we want to predict.
    '''
    
    # DF where we'll append predictions as we generate them. Copying 
    # most recent `fit_window` days from stock_data so that we 
    # have a basis for predictions.
    df = stock_data.iloc[-fit_window:]

    for _ in range(window):        
        # Scale data and reshape for model
        #
        # Still using fit_window here because we only want
        # most recent fit_window days for making prediction
        data = prep_data_for_pred(df.iloc[-fit_window:], stock_data, fit_window)
        
        # Get prediction
        pred = model.predict(data)
        
        # Inverse transform and reshape predicted price
        pred = rescale_pred(pred, stock_data)
        pred = pred.reshape(-1,)
        
        # Get date for adding new row to index
        pred_date = df.iloc[-1].name + timedelta(days=1)
        
        # Add predictions to df
        df = df.append(pd.DataFrame({'close': pred}, index=[pred_date]))
        
        # Fit model with added data (predictions)
        X, y = prep_data_for_fitting(df, fit_window)
        model.fit(X, y, epochs=10, shuffle=False, batch_size=1, verbose=0)
        
    # After for loop, last row in df will be the prediction we want.
    pred_to_return = df.iloc[-1]
    
    # Calculate pred_return, sharpe ratio, date of prediction (just to 
    # be transparent for user)
    df['return'] = df['close'].pct_change()

    # Calc predicted return
    pred_return = df.iloc[-1]['return'] * 100
    pred_return = "{0:.3f}".format(pred_return)
    
    # Calc sharpe ratio
    sharpe_ratio = (df['return'].mean() * 252) / (df['return'].std() * np.sqrt(252))
    sharpe_ratio = "{0:.3f}".format(sharpe_ratio)
    
    # Get predicted date
    predicted_date = pred_to_return.name
    
    return pred_return, sharpe_ratio, predicted_date



def get_portfolio_predictions(tickers, window=30):
    '''
    *** Entry point into module.
    '''
    
    predicted_values = dict.fromkeys(tickers)
    
    for ticker in tickers:
        # If we had saved models, we could check for those here
        # and update them if they exist.
        ### Testing saved models
        # model, data = create_model_and_dataset(ticker)
        model, data = get_model_and_data(ticker)
        print(model)
        
        pred_return, sharpe_ratio, predicted_date = predicted_portfolio_metrics(model=model, stock_data=data, window=window)
        
        val_dict = {
            'predicted_return': pred_return,
            'sharpe_ratio': sharpe_ratio,
            'predicted_date': predicted_date
        }
        
        predicted_values[ticker] = val_dict
        
    return predicted_values



if __name__ == '__main__':
    '''
    Use `argparse` to get CLI flags from user.
    '''
    # In case testing, create timestamp at runtime.
    start = datetime.now()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default=False, help='Test')
    parser.add_argument('--portfolio', dest='portfolio', help='Tickers (comma separated)')
    args = parser.parse_args()

    # Get tickers from parsed args
    # ticker_strs = args.portfolio
    # tickers = [tick.strip() for tick in ticker_strs.split(',')]
    tickers = args.portfolio

    # Main/top entry to ML part of the module
    user_portfolio_metrics = get_portfolio_predictions(tickers)
    
    # Log predicted data to console
    print(user_portfolio_metrics)
    
    # If testing, print the time
    if args.test:
        print(f"Run time: {datetime.now()-start}")
