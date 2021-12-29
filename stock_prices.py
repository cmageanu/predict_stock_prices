#!/usr/bin/env python

import argparse
import pandas_datareader as pdr
import pandas as pd

from datetime import datetime
import datetime as dt

from pandas import concat

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import pickle
from pathlib import Path

import os
import sys
import yaml

import pandas_market_calendars as mcal


def get_args():
    '''
    parse the arguments passed to the script
    '''
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(help='action', dest='action')
    train_parser = subparsers.add_parser('train')
    predict_parser = subparsers.add_parser('predict')
    
    train_parser.add_argument("start_date", help="start date of the training data period in YYYY-MM-DD format")
    train_parser.add_argument("end_date", help="end date of the training data period in YYYY-MM-DD format;\
                                                the training period must be at least 2 years")
    train_parser.add_argument("predict_period", help="predicted future period 1..28 days", type=int, choices=range(1, 29), metavar='[1-28]')
    train_parser.add_argument("symbols", help="one or more symbols", nargs='+')
    
    predict_parser.add_argument('-d','--dates', action='append', help='<Required> list of dates to predict prices in YYYY-MM-DD fromat', required=True)
    predict_parser.add_argument('-s','--symbols', action='append', help='<Required> list of symbols', required=True)
    
    args = parser.parse_args()
    
    if args.action is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return args


def df_to_lagged_features(df, n_in=1, n_out=0, dropnan=True):
    """
    Creates features for a data frame suitable for supervised learning.
    Arguments:
        df: Pandas dataframe.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series with lagged columns backwards and forwards suitable for supervised learning.
    """

    lagged_columns, lagged_column_names = list(), list()
    df_column_names = df.columns
	# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        lagged_columns.append(df.shift(i))
        lagged_column_names += [('%s(t-%d)' % (column_name, i)) for column_name in df_column_names]

    # add current value (moment t)
    lagged_column_names += [ '%s(t)' % (column_name) for column_name in df_column_names]
    lagged_columns.append(df)
    
    # future moments if any (t+1, ... t+n)
    if n_out >= 1:
        for i in range(1, n_out+1):
            lagged_columns.append(df.shift(-i))
            lagged_column_names += [('%s(t+%d)' % (column_name, i)) for column_name in df_column_names]
    
    # put it all together
    agg = concat(lagged_columns, axis=1)
    agg.columns = lagged_column_names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_yahoo_symbol_data(symbol):
    """
    Retrieves daily trading data from Yahoo Finance. Saves it as file data/symbol.csv, overwriting any existing data.
    Arguments:
        symbol: string, stock symbol i.e GOOG, TSLA
        start_date: start date of the retrieved data set as a datetime object
    Returns:
        Pandas DataFrame with the following columns: High, Low, Open, Close, Volume, Adj Close
        All available data for the respective symbol is retrieved, if any.
    """    
    try:
        symbol_data = pdr.get_data_yahoo(symbols=symbol, start=datetime(1900, 1, 1))
        # pickle data
        pickle.dump(symbol_data, open('data/{0}.pickle'.format(symbol), 'wb'))
    except:
        print('error encountered on downloading symbol {0}; skipping it'.format(symbol))
        symbol_data = None
    
    return symbol_data


def get_symbol_data(symbol, start_date='2000-01-01'):
    """
    Retrieves daily trading data from Yahoo Finance. Saves it as file data/symbol.csv.
    If data/symbol.csv exists and was created today, then no downloading occurs.
    Arguments:
        symbol: string, stock symbol i.e GOOG, TSLA
        start_date: start date of the retrieved data set, if format YYYY-MM-DD
    Returns:
        Pandas DataFrame with the following columns: High, Low, Open, Close, Volume, Adj Close
        If the symbol data does not extend back in time to start_date, then all available data for the respective symbol is retrieved, if any.
    """

    today = datetime.now().date()
    symbol_filename = 'data/{0}.pickle'.format(symbol)
       
    # if symbol_filename does not exist or is different from today, then download symbol data and save it as data/{symbol}.csv
    if not Path(symbol_filename).is_file():
        symbol_data = get_yahoo_symbol_data(symbol)
    else:
        symbol_filedate = datetime.fromtimestamp(os.path.getctime(symbol_filename)).date()
        if not symbol_filedate == today:
            symbol_data = get_yahoo_symbol_data(symbol)
        else:
            symbol_data = pickle.load(open(symbol_filename, 'rb'))
    
    if symbol_data is not None:
        symbol_data = symbol_data.loc[start_date:]
    
    return symbol_data


def next_trading_day(date):
    """
    Returns True if date id a NYSE trading day; otherwise returns False
    Arguments:
        date: type:string, format: YYYY-MM-DD
    Returns:
        boolean
    """
    # nt_days is of type pandas.core.indexes.datetimes.DatetimeIndex
    nt_days = nyse.valid_days(date, pd.to_datetime(date) + dt.timedelta(days=7))
    
    # the first element in nt_days is the date argument if date is a trading day
    # in this case, we need to unpack and return the second element in nt_days
    # otherwise, the next trading day is the first element in nt_days
    if nt_days[0].to_pydatetime().strftime('%Y-%m-%d') == date:
        return nt_days[1].to_pydatetime().strftime('%Y-%m-%d')
    else:
        return nt_days[0].to_pydatetime().strftime('%Y-%m-%d')
        
    
def is_trading_day(date):
    """
    Returns next NYSE trading day relative to the argument date received
    Arguments:
        date: type:string, format: YYYY-MM-DD 
    Returns:
        next NYSE trading day as a string, in format: YYYY-MM-DD
    """

    if len(nyse.valid_days(date, date)) == 1:
        return True
    else:
        return False
        
    
def train_model(start_date, end_date, predict_period, symbol):
    """
    Trains a linear model for a symbol.
    Arguments:
        start_date: start date of the training data set in format YYYY-MM-DD
        end_date: end date of the training data set in format YYYY-MM-DD
        symbol: string, stock symbol i.e GOOG, TSLA
    Outputs:
        saves a linear model in pickle format as models/symbol_model.pickle
        saves metadata about the model as models/symbol_model.metadata
    Returns:
        None
    """
    
    n_in = 20
    n_out = predict_period
    
    symbol = symbol.upper()
    symbol_data = get_symbol_data(symbol)
    symbol_data = symbol_data.loc[start_date:end_date]
    
    # if there is not enough data (2 years of trading days), print a warning message and skip this symbol
    if len(symbol_data) < 504:
        print('No predictions were generated for symbol {} and the requested training period as not enough data found'.format(symbol))
        return
    
    print('Generating predictions for symbol {}'.format(symbol))

    symbol_data = symbol_data.rename(columns={'High':'h', 'Low':'l', 'Open':'o', 'Close':'c', 'Volume':'v', 'Adj Close':'ac'})
    
    # create lagged features
    dflag = df_to_lagged_features(symbol_data.loc[:, ['ac']], n_in, n_out)

    # drop columns ac(t+1)..ac(t+2) from dflag to create X features
    columns_to_drop = [ 'ac(t+{0})'.format(i) for i in range(1, n_out+1) ]
    X = dflag.drop(columns_to_drop, 1)

    # predicted values are all non X columns
    y = dflag.drop(X.columns, 1)

    # create test/train datasets
    test_size = n_out

    not_for_train_size = n_out + n_in + test_size

    X_train = X.iloc[:len(X)-not_for_train_size]
    y_train = y.iloc[:len(y)-not_for_train_size]


    # fit and predict a multiclass linear model
    model = LinearRegression(normalize=True)
    model = MultiOutputRegressor(model)
    model.fit(X_train, y_train)
    
    # save the model
    pickle.dump(model, open( "models/{}_model.pickle".format(symbol), "wb" ) )
    
    # save the model metadata
    metadata = {'start_date': start_date, 'end_date': end_date, 'predicted_period': predict_period}
    with open("models/{}_metadata.yaml".format(symbol), "w") as predictions_filename:
        yaml.dump(metadata, predictions_filename)
        

    # save the predicted values
    # we need to predict on the end_date, the last value of symbol_data, which is lost when we shift columns to created lagged features.
    # to recover it, recreate lagged features with preserving NA values

    # create lagged features
    dflag = df_to_lagged_features(symbol_data.loc[:, ['ac']], n_in, n_out, dropnan=False)

    # drop columns ac(t+1)..ac(t+2) from dflag to create X features
    columns_to_drop = [ 'ac(t+{0})'.format(i) for i in range(1, n_out+1) ]
    X = dflag.drop(columns_to_drop, 1)
    
    predicted_values = model.predict(X[-1:])[0].tolist()
    predicted_days = []
    current_trading_day = X[-1:].index.to_pydatetime()[0].strftime('%Y-%m-%d')
    for _ in range(predict_period):
        ntd = next_trading_day(current_trading_day)
        predicted_days.append(ntd)
        current_trading_day = ntd
    
    predictions = dict(zip(predicted_days, predicted_values))
    with open("models/{}_predictions.yaml".format(symbol), "w") as predictions_filename:
        yaml.dump(predictions, predictions_filename)
    


def predict(dates, symbols):
    """
    Prints predicted values for symbols an avilable predicted dates.
    The predicted price symbols are created in models/{symbol}_predictions.yaml files during the train phase.
    Arguments:
        dates: list of dates; a date is a string in YYYY-MM-DD format
        symbols: list of symobls; a symbol is a string i.e GOOG, TSLA
    Outputs:
        For each symbol and each date it prints predicted prices if found.
        Otherwise it prints an error message
    Returns:
        None
    """
    
    for symbol in sorted(symbols):
        print(symbol)
        predictions_filename = "models/{}_predictions.yaml".format(symbol)
        try:
            predictions = yaml.load(open(predictions_filename, 'r'), Loader=yaml.Loader)
        except:
            print('    Could not find predictions file for symbol {}'.format(symbol))
            continue
        
        for date in dates:
            if date in predictions.keys():
                print('    {}  {:7.2f}'.format(date, predictions[date]))
            else:
                print('    {} no prediction could be found for this date'.format(date))
        


def main():

    # main entry point of the program
    
    args = get_args()
    action, symbols = args.action, args.symbols
    if action == 'train':
        start_date, end_date, predict_period = args.start_date, args.end_date, args.predict_period
        
        # check the start_date is in YYYY-MM-DD format 
        try:
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        except:
            print('Please check the start date {0} is in YYYY-MM-DD format'.format(start_date))
            sys.exit(1)
        
        # check the start_date is an NYSE trading date
        if not is_trading_day(start_date):
            print('Start date must be an NYSE trading date')
            sys.exit(1)
 
        # check the end_date is in YYYY-MM-DD format
        try:
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except:
            print('Please check the end date {0} is in YYYY-MM-DD format'.format(end_date))
            sys.exit(1)
        
        # check the end_date is an NYSE trading date
        if not is_trading_day(end_date):
            print('End date must be an NYSE trading date')
            sys.exit(1)
 
        # check the training period end_date - start_date is at least two years:
        if (end_date_dt - start_date_dt).days < 730:
            print('The difference between end_date and start_date must be at least 2 years (730 days).')
            sys.exit(1)
        
        for symbol in symbols:
            train_model(start_date, end_date, predict_period, symbol)
    else:
        dates, symbols = args.dates, args.symbols
        # check all dates are in YYYY-MM-DD format
        for date in dates:
            try:
                date = datetime.strptime(date, '%Y-%m-%d')
            except:
                print('Please input the date {0} is in YYYY-MM-DD format'.format(date))
                sys.exit(1)
                
        predict(dates, symbols)


if __name__ == '__main__':
    nyse = mcal.get_calendar('NYSE')
    main()
    