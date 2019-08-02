import os
import argparse
import asyncio

from core.CryptoCompare import *
from core.neuralnet import *
from core.preprocessing import *
from core.tools import db_to_csv, update_from_env