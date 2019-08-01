import json
import asyncio
import time
import pandas as pd

from threading import Timer
from typing import List
from tools import adel, aput, apost, aget, time_left_in_month

class CryptoCompareAPI:
  """
  A class for making API requests to CryptoCompare API
  """
  def __init__(self, config):
    """
    Initializes a new class for making API requests

    Args:
      config (dict) - the config to initialize this repo with
    """
    self.apiKey     =     config['apiKey']
    self.urlBase    =    config['urlBase']
    self.endpoints  =  config['endpoints']
    self.limits     = config['rateLimits']
    self.authHeader = {
      'authorization' : f'ApiKey {self.apiKey}'
    }
    self.callsMade = {
      'second' : 0,
      'minute' : 0,
      'hour'   : 0,
      'day'    : 0,
      'month'  : 0,
    }
    self.callsRemain = self.limits.copy()

    ms = self.milliseconds()

    self.lastCall = ms

    timers = {
      'second' : {
      },
      'minute' : {
      },
      'hour' : {
      },
      'day' : {
      },
      'month' : {
      }
    }
    self.timers = timers
    self.start_timers()

  def start_timers(self):
    """
    Starts timers to reset statistics
    """
    for timer in self.timers:
      self.timers[timer]['started'] = self.milliseconds()
      if timer == 'second': duration = 1000 
      elif timer == 'minute': duration = 60 * 1000
      elif timer == 'hour': duration = 60 * 60 * 1000
      elif timer == 'month': duration = time_left_in_month()
      self.timers[timer]['duration'] = duration
      self.timers[timer]['thread'] = Timer(duration / 1000, lambda : self.reset_timer(timer))
      self.timers[timer]['thread'].start()
  
  def stop_timers(self):
    """
    Stops timers to reset statistics
    """
    for timer in self.timers:
      self.timers[timer]['thread'].cancel()
  
  def reset_timer(self, timer: str):
    """
    Resets timer timer

    Args:
      timer (str) - the timer to reset
    """
    self.callsMade[timer] = 0
    self.callsRemain[timer] = self.limits[timer]
    self.timers[timer]['started'] = self.milliseconds()
    if timer == 'second': duration = 1000 
    elif timer == 'minute': duration = 60 * 1000
    elif timer == 'hour': duration = 60 * 60 * 1000
    elif timer == 'month': duration = time_left_in_month()
    self.timers[timer]['duration'] = duration
    self.timers[timer]['thread'] = Timer(duration / 1000, lambda : self.reset_timer(timer))
    self.timers[timer]['thread'].start()

  def milliseconds(self):
    """
    Returns: the timestamp in milliseconds
    """
    return int(time.time() * 1000)
  
  def incr_call_counts(self):
    """
    Increments call counters
    """
    for timer in self.timers:
      self.callsMade[timer] += 1
      self.callsRemain[timer] -= 1

  async def throttle(self):
    """
    Throttles the request to ensure that rate limits are not hit
    """
    remains = self.timers['month']['duration'] - (self.milliseconds() - self.timers['month']['started'])
    elapsed = self.milliseconds() - self.lastCall
    call_rate = remains / self.callsRemain['month']
    if elapsed < call_rate: await asyncio.sleep(call_rate - elapsed)
  
  async def get_usage(self):
    """
    Requests usage statistics for this API
    """
    await self.throttle()
    response = await aget(self.urlBase + self.endpoints['usage'], headers = self.authHeader)
    self.incr_call_counts()
    self.lastCall = self.milliseconds()
    response.raise_for_status()
    js = response.json()
    return js['Data']

  async def fetch_daily_ohlcv(self, base: str, quote: str, limit: int = 2000, toTime: int = (time.time() * 1000)):
    """
    Gets the daily price change data from CryptoCompare API

    Arg:
      base (str) - the base symbol to get ohlcv data for
      quote (str) - the quote symbol to read ohlcv data as
      limit (int) - the number of days to look back in each request
      toTime (int) - the time to end the request

    Returns:
      df (DataFrame) - a dataframe that contains all historical price data
      about a given base/quote pair dating back limit number of days. 
    """
    df = pd.DataFrame(
      columns = [
        'timestamp',
        'symbol',
        'toSymbol',
        'close',
        'high',
        'low',
        'open',
        'volumeFrom',
        'volumeTo',
      ])
    await self.throttle()
    response = await aget(self.urlBase + self.endpoints['dailyOhlcv'], headers = self.authHeader, params = {
      'fsym'  :        base,
      'tsym'  :       quote,
      'limit' :       limit,
      'toT'   : toTime/1000,
    })

    self.incr_call_counts()
    self.lastCall = self.milliseconds()
    response.raise_for_status()
    js = response.json()
    for candle in js['Data']:
      df = df.append(
        {
          'timestamp'  :  candle['time']*1000,
          'symbol'     :                 base,
          'toSymbol'   :                quote,
          'close'      :      candle['close'],
          'high'       :       candle['high'],
          'low'        :        candle['low'],
          'open'       :       candle['open'],
          'volumeFrom' : candle['volumefrom'],
          'volumeTo'   :   candle['volumeto'],
        }, ignore_index = True)
    return df.set_index('timestamp')
  
  async def fetch_full_data(self, bases: List[str], quotes: List[str]):
    """
    Gets the full data for every coin in bases in terms of every quote in quotes

    Args:
      bases (List[str]) - the base coins to get data for
      quotes (List[str]) - the quote coins to gen data in
    
    Returns:
      df (DataFrame) - a dataframe containing the full history for every coin in 
      bases in terms of every coin in quotes
    """
    df = pd.DataFrame(
      columns = [
        'timestamp'         ,
        'symbol'            ,
        'toSymbol'          ,
        'price'             ,
        'lastVolume'        ,
        'lastVolumeTo'      ,
        'volumeDay'         ,
        'volumeDayTo'       ,
        'volume24Hour'      ,
        'volume24HourTo'    ,
        'openDay'           , 
        'highDay'           ,
        'lowDay'            ,
        'open24Hour'        ,
        'high24Hour'        ,
        'change24Hour'      ,
        'changePct24Hour'   ,
        'changeDay'         ,
        'changePctDay'      ,
        'supply'            ,
        'mktCap'            ,
        'totalVolume24Hr'   ,
        'totalVolume24HrTo' ,
      ])
    await self.throttle()
    response = await aget(self.urlBase + self.endpoints['fullData'], headers = self.authHeader, params = {
      'fsyms'   :  ','.join(bases)[:-1],
      'tosyms' : ','.join(quotes)[:-1],
    })
    self.incr_call_counts()
    self.lastCall = self.milliseconds()
    response.raise_for_status()
    js = response.json()

    for base in js['RAW']:
      for quote in js['RAW'][base]:
        d = js['RAW'][base][quote]
        suffix = d['FROMSYMBOL']
        df = df.append(
          {
            'timestamp'         :         self.lastCall,
            'symbol'            :       d['FROMSYMBOL'],
            'toSymbol'          :         d['TOSYMBOL'],
            'price'             :            d['PRICE'],
            'lastVolume'        :       d['LASTVOLUME'],
            'lastVolumeTo'      :     d['LASTVOLUMETO'],
            'volumeDay'         :        d['VOLUMEDAY'],
            'volumeDayTo'       :      d['VOLUMEDAYTO'],
            'volume24Hour'      :     d['VOLUME24HOUR'],
            'volume24HourTo'    :   d['VOLUME24HOURTO'],
            'openDay'           :          d['OPENDAY'], 
            'highDay'           :          d['HIGHDAY'],
            'lowDay'            :           d['LOWDAY'],
            'open24Hour'        :       d['OPEN24HOUR'],
            'high24Hour'        :       d['HIGH24HOUR'],
            'change24Hour'      :     d['CHANGE24HOUR'],
            'changePct24Hour'   :  d['CHANGEPCT24HOUR'],
            'changeDay'         :        d['CHANGEDAY'],
            'changePctDay'      :     d['CHANGEPCTDAY'],
            'supply'            :           d['SUPPLY'],
            'mktCap'            :           d['MKTCAP'],
            'totalVolume24Hr'   :   d['TOTALVOLUME24H'],
            'totalVolume24HrTo' : d['TOTALVOLUME24HTO'],
          }, ignore_index = True)
    return df.set_index('timestamp')

  async def fetch_coins(self, coins: List[str]):
    """
    Gets technical data for a list of coins

    Args:
      coins (List[str]) - list of coins to get technical data on

    Returns:
      df (DataFrame) - a dataframe containing technical information for each coin
      in coins
    """
    df = pd.DataFrame(
      columns = [
        'timestamp',
        'symbol',
        'totalCoinsMined',
        'blockNumber',
        'netHashesPerSecond',
        'blockReward',
        'blockTime'
      ])
    await self.throttle()
    response = await aget(self.urlBase + self.endpoints['coins'])
    self.incr_call_counts()
    self.lastCall = self.milliseconds()
    response.raise_for_status()
    js = response.json()
    data = filter(lambda s: s in coins, js['Data'])

    for symbol in data:
      d = data[symbol]
      df = df.append(
        {
          'timestamp'           :           self.lastCall,
          'symbol'              :                  symbol,
          'totalCoinsMined'     :    d['TotalCoinsMined'],
          'blockNumber'         :        d['BlockNumber'],
          'netHashesPerSecond'  : d['NetHashesPerSecond'],
          'blockReward'         :        d['BlockReward'],
          'blockTime'           :          d['BlockTime'],
        }, ignore_index = True)
    return df.set_index('timestamp')



