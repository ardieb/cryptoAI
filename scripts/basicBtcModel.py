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

df = pd.read_csv('../data/btcusd/AgrStats[2010-7-20][2017-12-27].csv', sep = ',')
x_train, y_train, x_test, y_test, y_prior, bases, window_size = window_transform(df, SEQUENCE_LENGTH, SPLIT_PERCENTAGE, ['Bitcoin Price'])

model = create_model(window_size, x_train.shape[-1], DROP_RATE, 1)
trained_model, training_time = fit_model(model, 'Adam', 'MSELoss', x_train, y_train, EPOCHS)

print(training_time)

y_pred, real_y_test, real_y_pred, fig = test_model(trained_model, x_test, y_test, bases, 'BTC/USD')
fig.show()



