# Predicting Stock Prices

This repository contains code for an exercise in predicting stock prices at Nasdaq

## Motivation for this project

This project is part of an assignment of the [Udacity Data Science nano degree program](https://www.udacity.com/course/data-scientist-nanodegree--nd025)


## Files

### Code

The code in the repository has been tested with Python 3.7.7

predict_stock_prices.py - main file, having a training and a predict interface.

The training interface that accepts a data range (start_date, end_date), a predict_period parameter (numbers of days in future that will be predicted) and a list of ticker symbols (e.g. GOOG, AAPL). 
It builds a model which is saved in the models/{symbol}_model.pickle file.
It also saves the predicted values in the models/{symbol}_predictions.yaml file.

The query interface that accepts a list of dates and a list of ticker symbols, and outputs the predicted stock prices for each of those stocks on the given dates.

Usage:

```
./stock_prices.py train --help
usage: stock_prices.py train [-h]
                             start_date end_date [1-28] symbols [symbols ...]

positional arguments:
  start_date  start date of the training data period in YYYY-MM-DD format
  end_date    end date of the training data period in YYYY-MM-DD format; the
              training period must be at least 2 years
  [1-28]      predicted future period 1..28 days
  symbols     one or more symbols

optional arguments:
  -h, --help  show this help message and exit


./stock_prices.py predict --help
usage: stock_prices.py predict [-h] -d DATES -s SYMBOLS

optional arguments:
  -h, --help            show this help message and exit
  -d DATES, --dates DATES
                        <Required> list of dates to predict prices in YYYY-MM-
                        DD fromat
  -s SYMBOLS, --symbols SYMBOLS
                        <Required> list of symbols
```

Example usage:

```
./stock_prices.py train 2013-01-02 2017-12-06 7 GOOG TSLA AAPL
Generating predictions for symbol GOOG
Generating predictions for symbol TSLA
Generating predictions for symbol AAPL

./stock_prices.py predict -d 2017-12-14 -d 2017-12-07 -s TSLA -s GOOG -s AAPL
AAPL
    2017-12-14    40.46
    2017-12-07    40.46
GOOG
    2017-12-14  1020.86
    2017-12-07  1019.29
TSLA
    2017-12-14    62.38
    2017-12-07    62.80

```


### Data files

These are stored in the data directory as {symbol}.pickle files
The data is sourced from Yahoo Finance with the pandas_datareader library.
Each {symbol}.pickle file contains all data available from Yahoo Finance in pandas DataFrame format as returned by pandas_datareader.
The data is sourced once a day only. If sourced once, the programs will use the {symbol}.pickle file from disk until the computer date changes.

Models, predicted values and models metadata files are stored in the models directory.

Examples:

```
ll models/GOOG_*
-rw-rw-r-- 1 cmageanu cmageanu   68 Dec 29 13:08 models/GOOG_metadata.yaml
-rw-rw-r-- 1 cmageanu cmageanu 4001 Dec 29 13:08 models/GOOG_model.pickle
-rw-rw-r-- 1 cmageanu cmageanu  230 Dec 29 13:08 models/GOOG_predictions.yaml

file models/GOOG_model.pickle
models/GOOG_model.pickle: data

cat models/GOOG_metadata.yaml
end_date: '2017-12-06'
predicted_period: 7
start_date: '2013-01-02'

cat models/GOOG_predictions.yaml
'2017-12-07': 1019.2860643090332
'2017-12-08': 1018.8306899447565
'2017-12-11': 1019.2195512713481
'2017-12-12': 1019.9163853261975
'2017-12-13': 1019.642753306974
'2017-12-14': 1020.8641797969664
'2017-12-15': 1023.7266636700152
```
### Other files

predict_stock_prices.ipynb - python notebook file; it contains data exploration in support of the methodology used in stock_prices.py; it takes up to 3 hours to execute; this notebook is included in html and markdown format for convenience

symbols.csv list of Nasdaq symbols downloaded from the Nasdaq website: https://www.nasdaq.com/market-activity/stocks/screener

### Modules required to run the code

argparse
datetime
matplotlib
numpy
os
pandas
pandas_datareader
pandas_market_calendars
pathlib
pickle
random
seaborn
sklearn
statistics
sys
xgboost
yaml

The code was tested in a conda created environment using conda 4.10.3 with Python 3.7.7 running on Ubuntu 20.04
