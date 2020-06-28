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