import talib
import pandas as pd
import numpy as np

from data_prerprocessing import config


def add_ma_features(tic_df):
    for t in range(1, 11):
        tic_df[f'ret{t}min'] = tic_df['close'].div(tic_df['open'].shift(t-1)).sub(1)
        tic_df['sma'] = talib.SMA(tic_df['close'])
        tic_df['5sma'] = talib.SMA(tic_df['close'], timeperiod=5)
        tic_df['20sma'] = talib.SMA(tic_df['close'], timeperiod=20)
        tic_df['sma_buy'] = np.where((tic_df['close'] < tic_df['5sma']), 1, 0)
    return tic_df


def add_features_to_csv(file_path, file_nm_list, preprocessed_path):
    for file_nm in file_nm_list:
        df = pd.read_csv(file_path + file_nm, index_col=0)
        df['date'] = pd.to_datetime(df['date'])
        df['weekday'] = df['date'].dt.weekday / 6

        updated_df = add_ma_features(tic_df=df)
        updated_df['date'] = pd.to_datetime(updated_df['date'])
        updated_df = df.set_index(pd.DatetimeIndex(updated_df['date']))
        updated_df.replace([np.inf, -np.inf], 0, inplace=True)
        updated_df.interpolate(method='pad', limit_direction='forward', inplace=True)
        updated_df.to_csv(preprocessed_path + file_nm)


def grouping_df(preprocessed_path, file_nm_list):
    df_list, price_df = [], pd.DataFrame()
    for file_nm in file_nm_list:
        df = pd.read_csv(preprocessed_path + file_nm, index_col=0)
        df.drop(['date'], axis=1, inplace=True)
        comp_nm = df['Name'][0]
        price_df[f'{comp_nm}_close'] = df['close']
        df_list.append(df)
    return df_list, price_df






