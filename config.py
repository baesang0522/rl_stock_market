import os
import torch


class Config:
    S_DATA_PATH = "./data/stock/"
    C_DATA_PATH = "./data/crypto/"
    S_FILE_LIST = os.listdir(S_DATA_PATH)
    C_FILE_LIST = os.listdir(C_DATA_PATH)
    PP_DATA_PATH = "./preprocessed_data/"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    INDICATORS = ['open', 'high', 'low', 'close', 'volume', 'ToD', 'DoW',
                  'ret1min', 'ret2min', 'ret3min', 'ret4min', 'ret5min', 'ret6min',
                  'ret7min', 'ret8min', 'ret9min', 'ret10min', 'sma', '5sma', '20sma',
                  'bb_upper', 'bb_middle', 'bb_lower', 'bb_sell', 'bb_buy', 'bb_squeeze',
                  'mom', 'adx', 'mfi', 'rsi', 'trange', 'bop', 'cci', 'STOCHRSI', 'slowk',
                  'slowd', 'macd', 'macdsignal', 'macdhist', 'NATR', 'KAMA', 'MAMA',
                  'FAMA', 'MAMA_buy', 'KAMA_buy', 'sma_buy', 'maco', 'rsi_buy',
                  'rsi_sell', 'macd_buy_sell']


class DevConfig(Config):
    pass


class ProdConfig(Config):
    pass


config = {"dev": DevConfig(),
          "prod": ProdConfig()}







