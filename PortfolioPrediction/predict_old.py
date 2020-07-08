import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import os

class WeightSelector(Object):
    def __init__(self):
        # TODO
        pass

    def __repr__(self):
        # TODO
        # return self.clf.summary()
        pass

class TestModels(Object):
    def __init__(self, tickers, period):
        self.tickers = tickers
        self.period = period
        

def get_predictions(model, data_df, days_ahead):
    ''' 
    Since we're doing time-series predictions, we need to predict individually
    the days up to the one we want. If we're predicting 30 days ahead, then
    we first need to predict the return 29 days ahead, and for 29 days
    ahead we need the return for 28 days ahead, etc.
    
    `data_df` should be a *copy* of an existing DF, since we'll be adding
    rows using predicted data, which should not be saved in the original DF.
    '''
    
    assert data != []
    
    most_recent_return = data_df.iloc[-1]
    
    if days_ahead == 0:
        return 1 + most_recent_return
    else:
        return_pred = model.predict(most_recent_return)
        
        s = pd.Series()
        
        # Currently our columns are: close, lag_1_day, lag_return, short_window_ma,
        # long_window_ma, fast_vol, slow_vol.
        # We need to add all of these as a new row to data_df so that we can run
        # the model again to predict the following day.
        s['close'] = 
        
        return 1 + pred * get_predictions(model, data, days_ahead-1)

def calc_return(closing_price_1, closing_price_2):
    '''
    Same as `Series.pct_change` but can be used generally.
    
    `closing_price_2` is assume to be the more recent price.
    '''
    return (closing_price_2 - closing_price_1) / closing_price_1


most_recent = amzn_df.drop(columns=['return']).iloc[-1].values.reshape(1, -1)

# X_test_amzn.iloc[-1].values.reshape(1, -1)
get_predictions(mlp_amzn, most_recent, 10)