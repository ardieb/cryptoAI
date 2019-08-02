import sqlite3 as db
import csv

import asyncio
import concurrent.futures
import requests
import os

from typing import List
from datetime import datetime, timezone, timedelta


def update_from_env(d: dict, variables: List[str], inplace: bool = False):
  """
  Updates dictionary with environment variables

  Arguments:
    d              (dict): the dictionary to update
    variables (List[str]): the list of variables to update from the environment
    inplace        (bool): whether the update should occur inplace

  Returns:
    A new dictionary with the combined keys if inplace = False
  """
  new_keys = {}
  for var in variables:
    new_keys[var] = os.environ[var]

  if inplace:
    d.update(new_keys)
    return d

  for key in d:
    if key not in new_keys:
      new_keys[key] = d[key]

  return new_keys


def db_to_csv(source: str, target: str, table: str):
  """
  Converts a sqlite3 database table to a csv file

  Arguments:
    source (str) - the filename of the source database
    target (str) - the filename of the target csv file
    table  (str) - the table to selectr from the database
  """
  with csv.writer(open(target, 'w')) as writer:
    conn = db.connect(source)
    cursor = conn.cursor()
    query = f'SELECT * FROM {table}'

    for row in cursor.execute(query):
      writer.writerow(row)


def timestamp_floor(ts: int, how: str = 'day', unit: str = 'ms'):
  """
  Gets the floor of the timestamp in terms of how

  Arguments:
    ts (int) - the utc timestamp to floor
    how (str) - one of 'second', 'minute', 'hour', 'day', 'month'
    unit (str) - the unit the timestmap is in (one of 'ms' or 's')

  Returns:
    timestamp (int) - a unix timestamp in utc time floored using method `how`
  """
  dt = datetime.fromtimestamp(ts / 1000 if unit == 'ms' else ts,
                              tz = timezone.utc)
  if how == 'second':
    new_dt = datetime(year = dt.year, month = dt.month, day = dt.day,
                      hour = dt.hour, minute = dt.minute, second = dt.second,
                      tzinfo = timezone.utc)
  elif how == 'minute':
    new_dt = datetime(year = dt.year, month = dt.month, day = dt.day,
                      hour = dt.hour, minute = dt.minute, tzinfo = timezone.utc)
  elif how == 'hour':
    new_dt = datetime(year = dt.year, month = dt.month, day = dt.day,
                      hour = dt.hour, tzinfo = timezone.utc)
  else:
    new_dt = datetime(year = dt.year, month = dt.month, tzinfo = timezone.utc)

  timestamp = dt.replace(tzinfo = timezone.utc).timestamp()
  return int(timestamp * 1000 if unit == 'ms' else timestamp)


def noblock(f):
  """
  A decorator for performing an asyncronous request

  Arguments:
    f (func) - the function to wrap
  """

  async def wrapper(*args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers = 20) as executor:
      loop = asyncio.get_event_loop()
      response = await loop.run_in_executor(executor,
                                            lambda: f(*args, **kwargs))
      return response

  return wrapper


@noblock
def aget(url, **kwargs):
  """
  Async version of get
  """
  return requests.get(url, **kwargs)


@noblock
def apost(url, **kwargs):
  """
  Async version of post
  """
  return requests.post(url, **kwargs)


@noblock
def aput(url, **kwargs):
  """
  Async version of put
  """
  return requests.put(url, **kwargs)


@noblock
def adel(url, **kwargs):
  """
  Async version of del
  """
  return requests.delete(url, **kwargs)


def time_left_in_month():
  """
  Computes the milliseconds to the next month
  """
  now = datetime.now()

  year = now.year
  month = now.month
  day = now.day
  hour = now.hour
  minute = now.minute
  second = now.second

  M = (month + 1) % 12
  Y = year + 1 if M == 1 else year
  return int((datetime(Y, M, 1) - datetime(year, month, day, hour = hour,
                                           minute = minute,
                                           second = second)).total_seconds() * 1000)
