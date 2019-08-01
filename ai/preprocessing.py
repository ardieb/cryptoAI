"""
Preprocessing of data to make it more manageable
"""
import pandas as pd
import numpy as np
from torch import tensor

from tools import timestamp_floor

def split(*datasets):
  """
  Splits the dataframes in datasets into dataframes indexed by common base and 
  quote symbols

  Args:
    datasets (DataFrames) - a series of dataframes to split

  Returns:
    split (DataFrames) - a dictionary of split dataframes indexed by base and quotes
  """
  split = {}
  for df in datasets:
    symbols = pd.unique(df['symbol'])
    if 'toSymbol' in df.columns:
      to_symbols = pd.unqiue(df['toSymbol'])
      for symbol in symbols:
        for to_symbol in to_symbols:
          subframe = df[df['symbol'] == symbol and df['toSymbol'] == to_symbol]
          if (symbol, to_symbol) in split:
            split[(symbol, to_symbol)] = split[(symbol, to_symbol)].merge(subframe, how = 'outer')
          else:
            split[(symbol, to_symbol)] = subframe
    else:
      for symbol in symbols:
        subframe = df[df['symbol'] == symbol]
        for (f, t) in split:
          if symbol == f:
            split[(f,t)] = split[(f, t)].merge(subframe, how = 'outer')
  return split

def squeeze(dataset, how: str = 'day'):
  """
  Squeezes the data in dataset by close timestamps

  Args:
    dataset (DataFrame) - the data to squeeze
    how           (str) - one of 'second', 'minute', 'hour', 'day', 'month' (default day)

  Returns:
    dataset (DataFrame) - a dataframe where the indexes are squeezed together by closely related timestamps
    determined by parameter how
  """
  return dataset.groupby(by = lambda ts: timestamp_floor(ts, how = how))

def window_transform(dataset, seq_len: int, train: float, test: float, targets: List[str]):
  """
  Transforms a dataset through a sliding window trasnformation. Takes a dataset
  of shape n x m and transforms it to n - w, d, m where w is the window size,
  d is the number of samples in each window, m is the number of features

  Args:
    dataset         (Any) - the dataset to load (n, m) where n is the number of samples and m is the number of features
    seq_len         (int) - the length of the sequence of each window, 
    train         (float) - the percentage of data to use for training
    test          (float) - the percentage of data to use for testing
    targets   (List[str]) - the list of columns to use as validation data

  Returns:
    x_train           (ndarray) - training x data (n - w * test, d, m) shape
    y_train           (ndarray) - training y data (n - w,) shape 
    x_test            (ndarray) - testing x data (n - w - (n - w * test), d, m) shape
    y_test            (ndarray) - testing y data (n - w - (n - w * test), ) shape
    y_prior           (ndarray) - validator data of prior sample (n - w - (n - w * test), ) shape
    unormalized_bases (ndarray) - un-normalized data that is used to test training models (n - w - (n - w * test), ) shape
    window_size           (int) - integer representing how many samples the model can look at concurrently
  """
  n, m = dataset.shape
  for x in range(0, n):
    for y in range(0, m):
      if dataset.loc[x,y] == 0: dataset.loc[x,y] = dataset.loc[x-1, y]

  target_indicies = [dataset.columns.get_loc(target) for target in targets]

  data = dataset.tolist()

  res = []

  for index in range(len(data) - seq_len):
    res.append(data[index: index + seq_len])
  
  d0 = np.array(res)
  dr = np.zeros_like(res)
  dr[:, 1:, :] = d0[:, 1:, :] / d0[:, 0:1, :] - 1

  windows, samples, features = dr.shape
  split_line = int(round(test * windows))
  unormalized_bases = d0[split_line:windows+1, 0:1, target_indicies]

  training_data = dr[:split_line, :]
  testing_data  = dr[split_line:, :]
  np.random.shuffle(training_data)
  
  
  x_train = training_data[:, :-1]
  y_train = training_data[:, -1, target_indicies]

  x_test  = testing_data[:, :-1]
  y_test  = testing_data[:, -1, target_indicies]

  y_prior = testing_data[:, -2, target_indicies]
   
  window_size = seq_len - 1

  # use the previous window_size features to predict features in the final column
  return tensor(x_train), tensor(y_train), tensor(x_test), tensor(y_test), tensor(y_prior), tensor(unormalized_bases), window_size









  

