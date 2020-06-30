import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

from pathlib import Path
import os

class WeightSelector():
    def __init__(self, tickers, period=30):
        # TODO
        self.tickers = tickers
        try:
            self.tickers != []
        except EmptyStockList:
            print('No stocks provided.')
        # `period` is how far ahead we want to predict
        #
        # Implicit is that we will reevaluate on today+period,
        # so we're concerned with predicting the return on today+period.
        self.period = period
        try:
            self.period > 0
        except ZeroPeriod:
            print('Period must be greater than 1')
        self.today = datetime.today()

    def __repr__(self):
        # TODO
        # return self.clf.summary()
        pass

class TestModels():
    def __init__(self, tickers, period):
        self.tickers = tickers
        self.period = period

def get_trained_model():
    # TODO
    pass

def test():
    ws1 = WeightSelector(['AAPL', 'AMZN', 'NFLX'], 10)
    ws2 = WeightSelector([])

if __name__ == '__main__':
    test()