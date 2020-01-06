import os
import pandas as pd
import numpy as np
from pandas import concat


def load_data(data, n_in=1, n_out=1, lags=10, num_variate=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    # train test split
    array = agg.to_numpy()
    x = array[:, 0: lags]
    y = array[:,lags]
    train_size = int(train_size*len(array))
    x_train, y_train, x_val, y_val = x[0:train_size,:], y[0:train_size], x[train_size:,:], y[train_size:]
    x_train =  x_train.reshape(x_train.shape[0], lags//features, 
                                features)
    x_val =  x_val.reshape(x_val.shape[0], lags//features, 
                                features)
    return ((x_train, y_train), (x_val, y_val))


# upload dataset
path = 'AirPassengers.csv'
dataset = pd.read_csv(path, index_col=0, parse_dates=True)

# insert the lags accoring to requirement
lags = 12
num_variate = 1

((x_train, y_train), (x_val, x_val)) = load_data(dataset, lags, num_variate)


