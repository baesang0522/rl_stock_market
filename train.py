import os
import gym
import glob
import talib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import config as c
from gym import spaces
from tabnanny import verbose
from gat_policy import GATActorCriticPolicy
from custom_policy import CustomActorCriticPolicy
from stable_baselines3 import PPO, A2C, SAC, DDPG
from environment.custom_env import MultiStockTradingEnv

# config = c[os.environ["ENV"]]
config = c["prod"]

def add_feature(tic_df):
    # Returns in the last t intervals
    for t in range(1, 11):
        tic_df[f'ret{t}min'] = tic_df['close'].div(tic_df['open'].shift(t-1)).sub(1)

    # Simple Moving Average based features
    tic_df['sma'] = talib.SMA(tic_df['close'])
    tic_df['5sma'] = talib.SMA(tic_df['close'], timeperiod=5)
    tic_df['20sma'] = talib.SMA(tic_df['close'], timeperiod=20)
    tic_df['bb_upper'], tic_df['bb_middle'], tic_df['bb_lower'] = talib.BBANDS(tic_df['close'], matype=talib.MA_Type.T3)
    tic_df['bb_sell'] = (tic_df['close'] > tic_df['bb_upper']) * 1
    tic_df['bb_buy'] = (tic_df['close'] < tic_df['bb_lower']) * 1
    tic_df['bb_squeeze'] = (tic_df['bb_upper'] - tic_df['bb_lower']) / tic_df['bb_middle']
    tic_df['mom'] = talib.MOM(tic_df['close'], timeperiod=10)
    tic_df['adx'] = talib.ADX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=10)
    tic_df['mfi'] = talib.MFI(tic_df['high'], tic_df['low'], tic_df['close'], tic_df['volume'], timeperiod=10)
    tic_df['rsi'] = talib.RSI(tic_df['close'], timeperiod=10)

    tic_df['trange'] = talib.TRANGE(tic_df['high'], tic_df['low'], tic_df['close'])

    tic_df['bop'] = talib.BOP(tic_df['open'], tic_df['high'], tic_df['low'], tic_df['close'])
    tic_df['cci'] = talib.CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
    tic_df['STOCHRSI'] = talib.STOCHRSI(tic_df['close'], timeperiod=14, fastk_period=14,
                                        fastd_period=3, fastd_matype=0)[0]
    slowk, slowd = talib.STOCH(tic_df['high'], tic_df['low'], tic_df['close'], fastk_perid=14, slowk_period=3,
                               slowk_matype=0, slowd_period=3, slowd_matype=0)
    macd, macdsignal, macdhist = talib.MACD(tic_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    tic_df['slowk'], tic_df['slowd'], tic_df['macd'], tic_df['macdsignal'], tic_df['macdhist'] = \
        slowk, slowd, macd, macdsignal, macdhist

    tic_df['NATR'] = talib.NATR(tic_df['high'].ffill(), tic_df['low'].ffill(), tic_df['close'].ffill())
    tic_df['KAMA'] = talib.KAMA(tic_df['close'], timeperiod=10)
    tic_df['MAMA'], tic_df['FAMA'] = talib.MAMA(tic_df['close'])
    tic_df['MAMA_buy'] = np.where((tic_df['MAMA'] < tic_df['FAMA']), 1, 0)
    tic_df['KAMA_buy'] = np.where((tic_df['close'] < tic_df['KAMA']), 1, 0)
    tic_df['sma_buy'] = np.where((tic_df['close'] < tic_df['5sma']), 1, 0)
    tic_df['maco'] = np.where((tic_df['5sma'] < tic_df['20sma']), 1, 0)
    tic_df['rsi_buy'] = np.where((tic_df['rsi'] < 30), 1, 0)
    tic_df['rsi_sell'] = np.where((tic_df['rsi'] > 70), 1, 0)
    tic_df['macd_buy_sell'] = np.where((tic_df['macd'] < tic_df['macdsignal']), 1, 0)
    return tic_df


def adding_technical_indicator(file_path=config.S_DATA_PATH, file_list=config.S_FILE_LIST):
    dfs = pd.DataFrame()
    num_assets = 0
    names = []
    for file_nm in file_list:
        df = pd.read_csv(file_path + file_nm)
        df['datetime'] = pd.to_datetime(df['datetime'])
        name = df['name'].iloc[0]
        names.append(name)
        df['ToD'] = df['datetime'].dt.hour + df['datetime'].dt.minute/60























