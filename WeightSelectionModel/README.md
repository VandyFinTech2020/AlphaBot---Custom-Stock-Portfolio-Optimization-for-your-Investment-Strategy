# Stock portfolio prediction module

## Overview

Todo

## How to use

The code can be imported as a module or run stand-alone from the command line. 
When run from the command line, it accepts the flag `--portfolio` followed by a 
string of stock tickers, for example:

```
#prompt> python predict.py --portfolio 'AAPL, NFLX, GOOGL, TSLA'
```

This will return (log to the console) a dictionary of keys 'predicted_return', 
'sharpe_ratio' and 'predicted_date'.

The module can also be imported. `import predict` to load, then 

```
my_results = predict.get_portfolio_predictions(list_of_tickers)
```

to get prediction results.

### Testing

To show the running time at the command line, pass the `--test` flag, e.g.

```
#prompt> python predict.py --portfolio 'AAPL, NFLX, GOOGL, TSLA --test
```

(I considered doing more with this but couldn't think of what else might be needed.)
