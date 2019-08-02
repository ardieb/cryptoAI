import os
import argparse
import torch
import asyncio
import pandas as pd

from json import load

from core.CryptoCompare import *
from core.neuralnet import *
from core.preprocessing import *
from core.tools import db_to_csv, update_from_env

SEQUENCE_LENGTH = 25 # days
SPLIT_PERCENTAGE = 0.75 # x100%
DROP_RATE = 0.2
EPOCHS = 1000

with load(open('../cfg.json')) as cfg:
  cfg = update_from_env(cfg, ['CC_API_KEY'])

parser = argparse.ArgumentParser()
parser.add_argument('sources', metavar = 'S', nargs = '+', action = 'store',
                    help = 'the data sources to use', type = str)

sources = parser.parse_args().sources

dfs = split([pd.read_csv(source, sep = ',') for source in sources if os.path.isfile(source)])
df = dfs[('btc', 'usd')]
df = squeeze(df, how = 'day')

(x_train, y_train,
 x_test, y_test,
 y_prior, bases,
 window_size) = window_transform(df, SEQUENCE_LENGTH, SPLIT_PERCENTAGE, ['Price'])

model = create_model(window_size, DROP_RATE, 'linear', 1)
model = fit_model(model, 'Adam', x_train, y_train, EPOCHS)






