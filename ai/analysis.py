import torch
import pandas as pd
import asyncio
import argparse

from threading import Timer

from api import CryptoCompareAPI
from neuralnet import create_model, fit_model, test_model, price_change, binary_price, calculate_statistics
from preprocessing import window_transform, split, squeeze

async def collect_data(base: str, quote: str):
  """
  Continously collects data from Cypto Compare API
  """

