import os
import torch


class Config:
    S_DATA_PATH = "./data/stock/"
    C_DATA_PATH = "./data/crypto/"
    S_FILE_LIST = os.listdir(S_DATA_PATH)
    C_FILE_LIST = os.listdir(C_DATA_PATH)
    PP_DATA_PATH = "./preprocessed_data/"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DevConfig(Config):
    pass


class ProdConfig(Config):
    pass


config = {"dev": DevConfig(),
          "prod": ProdConfig()}







