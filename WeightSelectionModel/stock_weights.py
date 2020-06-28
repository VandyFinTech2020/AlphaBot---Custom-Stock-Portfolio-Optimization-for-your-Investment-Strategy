#!/usr/bin/env python
# coding: utf-8

# # Comparison of regression models in predicting next-day stock returns

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta
import os
from newsapi.newsapi_client import NewsApiClient
import alpaca_trade_api as tradeapi


import matplotlib.pyplot as plt

import hvplot.pandas



# Set News API Key
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

# Set Alpaca API key and secret
alpaca_api_key = os.getenv("ALPACA_API_KEY_ID")
alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")

api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, api_version='v2')



# TODO
# Selecting the tickers can be a stand-alone function
# Set the ticker
tickers = ["AAPL", "AMZN", "GOOGL", "NFLX"]

# Set timeframe to '1D'
timeframe = '1D'

# Get current date and the date from one month ago
current_date = date.today()
past_date = date.today() - timedelta(weeks=52)

df = pd.DataFrame()

# Get historical data for AAPL
for tick in tickers:
    tmp_df = api.get_barset(
        tick,
        timeframe,
        limit=365,
        start=current_date,
        end=past_date,
        after=None,
        until=None,
    ).df
    tmp_df = tmp_df.droplevel(0, axis=1)
    tmp_df.drop(columns=['open', 'high', 'low', 'volume'], inplace=True)
    tmp_df.rename({'close': tick}, axis=1, inplace=True)
    df = pd.concat([df, tmp_df], axis=1)


df.index = df.index.date

# Set global variables for consistent testing
short_window = 20
long_window = 100


# ## AAPL

# ### Create AAPL df

aapl_df = df.loc[:, 'AAPL'].to_frame()
aapl_df.rename(columns={'AAPL': 'close'}, inplace=True)
aapl_df['lag_1_day'] = aapl_df['close'].shift()
aapl_df.dropna(inplace=True)


# aapl_df.close.hvplot(subplots=True, figsize=(12,8))


aapl_df["return"] = aapl_df["close"].pct_change()
aapl_df.dropna(inplace=True)
 


short_window_aapl = 20
long_window_aapl = 100

# Exponentially weighted moving average
aapl_df["short_window_ma"] = aapl_df["close"].ewm(halflife=short_window_aapl).mean()
aapl_df["long_window_ma"] = aapl_df["close"].ewm(halflife=long_window_aapl).mean()

# Expoentially weighted volatility
aapl_df["fast_vol"] = aapl_df["return"].ewm(halflife=short_window_aapl).std()
aapl_df["slow_vol"] = aapl_df["return"].ewm(halflife=long_window_aapl).std()


aapl_df.dropna(inplace=True)


 
# Make copy of df for testing without `return` feature
aapl_df_copy = aapl_df.copy()

 
# X = aapl_df.close.values.reshape(-1, 1)
# X = aapl_df.close.values.reshape(-1, 1)
X_aapl = aapl_df.drop(columns=['lag_1_day'])
y_aapl = aapl_df.lag_1_day.values.reshape(-1, 1)


# ### Prep data for model fitting

from sklearn.model_selection import train_test_split

X_train_aapl, X_test_aapl, y_train_aapl, y_test_aapl = train_test_split(X_aapl, y_aapl, random_state=0, test_size=0.20)

from sklearn.preprocessing import StandardScaler

scaler_aapl = StandardScaler()

X_train_aapl_scaled, X_test_aapl_scaled = scaler_aapl.fit_transform(X_train_aapl), scaler_aapl.transform(X_test_aapl)


# ### AAPL with SVR

from sklearn.svm import SVR

svr_aapl = SVR()

svr_aapl.fit(X_train_aapl_scaled, y_train_aapl)

SVR_predictions_aapl = svr_aapl.predict(X_test_aapl_scaled)

SVR_predictions_aapl_df = pd.DataFrame({'predictions': SVR_predictions_aapl, 'actual': np.ravel(y_test_aapl)})

SVR_predictions_aapl_df.plot(figsize=(12,8))


from sklearn.metrics import mean_squared_error

SVR_MSE_aapl = mean_squared_error(y_test_aapl, SVR_predictions_aapl)

# print(f"SVR MSE: {SVR_MSE_aapl}")


# ### AAPL with Random Forest

from sklearn.ensemble import RandomForestRegressor

rfr_aapl = RandomForestRegressor(max_depth=2, random_state=0)

rfr_aapl.fit(X_train_aapl_scaled, y_train_aapl)

RF_predictions_aapl = rfr_aapl.predict(X_test_aapl_scaled)

RF_predictions_aapl_df = pd.DataFrame({'actual': np.ravel(y_test_aapl), 'predictions': RF_predictions_aapl})

RF_predictions_aapl_df.plot(figsize=(12,8))


RF_MSE_aapl = mean_squared_error(y_test_aapl, RF_predictions_aapl)

# print(f"SVR MSE: {SVR_MSE_aapl}\nRF_MSE: {RF_MSE_aapl}")


# ### AAPL with Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor

gbr_aapl = GradientBoostingRegressor(random_state=0)

gbr_aapl.fit(X_train_aapl_scaled, y_train_aapl)

GBR_predictions_aapl = gbr_aapl.predict(X_test_aapl_scaled)

GBR_predictions_aapl_df = pd.DataFrame({'actual': np.ravel(y_test_aapl), 'predictions': GBR_predictions_aapl})

# GBR_predictions_aapl_df.plot(figsize=(12,8))

GBR_MSE_aapl = mean_squared_error(y_test_aapl, GBR_predictions_aapl)

# print(f"GBR MSE: {GBR_MSE_aapl}\nRF_MSE: {RF_MSE_aapl}\nSVR_MSE: {SVR_MSE_aapl}")


# ## AMZN

# ### Prep data for model fitting

amzn_df = df.loc[:, 'AMZN'].to_frame()
amzn_df.rename(columns={'AMZN': 'close'}, inplace=True)
amzn_df['lag_1_day'] = amzn_df['close'].shift()
amzn_df.dropna(inplace=True)

amzn_df.head()

amzn_df["return"] = amzn_df["close"].pct_change()
amzn_df.dropna(inplace=True)
amzn_df.head() 

short_window_amzn = 20
long_window_amzn = 100

# Exponentially weighted moving average
amzn_df["short_window_ma"] = amzn_df["close"].ewm(halflife=short_window_amzn).mean()
amzn_df["long_window_ma"] = amzn_df["close"].ewm(halflife=long_window_amzn).mean()

# Expoentially weighted volatility
amzn_df["fast_vol"] = amzn_df["return"].ewm(halflife=short_window_amzn).std()
amzn_df["slow_vol"] = amzn_df["return"].ewm(halflife=long_window_amzn).std()

amzn_df.dropna(inplace=True)

X_amzn = amzn_df.drop(columns=['lag_1_day'])
y_amzn = amzn_df.lag_1_day.values.reshape(-1, 1)


from sklearn.model_selection import train_test_split

X_train_amzn, X_test_amzn, y_train_amzn, y_test_amzn = train_test_split(X_amzn, y_amzn, random_state=0, test_size=0.20)

from sklearn.preprocessing import StandardScaler

scaler_amzn = StandardScaler()

X_train_amzn_scaled, X_test_amzn_scaled = scaler_amzn.fit_transform(X_train_amzn), scaler_amzn.transform(X_test_amzn)

gbr_amzn = GradientBoostingRegressor(random_state=0)

gbr_amzn.fit(X_train_amzn_scaled, y_train_amzn)


GBR_predictions_amzn = gbr_amzn.predict(X_test_amzn_scaled)

GBR_predictions_amzn_df = pd.DataFrame({'actual': np.ravel(y_test_amzn), 'predictions': GBR_predictions_amzn})

GBR_predictions_amzn_df.plot(figsize=(12,8))



from sklearn.metrics import mean_squared_error

GBR_MSE_amzn = mean_squared_error(y_test_amzn, GBR_predictions_amzn)

# print(f"GBR1 MSE: {GBR_MSE}\nRF_MSE: {RF_MSE}\nSVR_MSE: {SVR_MSE}\nGBR2 MSE: {GBR2_MSE}")
# print(f"GBR2_MSE: {GBR_MSE_amzn}")

# print(gbr_amzn.score(X_test_amzn_scaled, y_test_amzn))
# print(gbr_aapl.score(X_test_aapl_scaled, y_test_aapl))
# print(rfr_aapl.score(X_test_aapl_scaled, y_test_aapl))
# print(svr_aapl.score(X_test_aapl_scaled, y_test_aapl))
