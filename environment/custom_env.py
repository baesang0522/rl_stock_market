import gym
import numpy as np


class MultiStockTradingEnv(gym.Env):
    """

    """
    metadata = {"render.modes": ["humans"]}

    def __init__(self, dfs, price_df, initial_amount, trade_cost, num_features, num_stocks,
                 window_size, frame_bound, scalers=None, tech_indicator_list=[], reward_scaling=1e-5):
        """
        :param dfs: 기업별 가상화폐별 raw 데이터를 담고있는 리스트
        :param price_df: 기업별 가상화폐별 종가를 담고있는 리스트
        :param initial_amount: 모델 training 시 사용될 초기 금액
        :param trade_cost: 거래비용(주식 사고팔때 발생하는 그것)
        :param num_features: 각 기업별 사용될 feature 들
        :param num_stocks: 트레이딩에 사용될 기업들
        :param window_size: 다음 Action을 할 때 고려될 이전 시간
        :param frame_bound: training/test에 사용될 start index, end index 범위. window size보단 커야함
        :param scalers: 각각의 주식을 scaling 할 scaler 리스트
        :param tech_indicator_list: 주식 기술지표 리스트
        :param reward_scaling: 각 주식 reward의 scaler
        """
        self.dfs = dfs
        self.price_df = price_df
        self.initial_amount = initial_amount
        self.trade_cost = trade_cost
        self.num_features = num_features
        self.num_stocks = num_stocks
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.scalers = scalers
        self.tech_indicator = tech_indicator_list
        self.reward_scaling = reward_scaling
        self.margin = initial_amount
        self.portfolio = [0] * num_stocks
        self.portfolio_value = 0
        self.reserve = initial_amount

    def reset(self, **kwargs):
        """
        모든 트래커를 리셋하는 메소드. 예를 들어 수익은 초기 금액(initial_amount)로 리셋됨.
        :return: 다음 관측값을 리턴
        """
        self._done = False
        self._current_tick = self._start_tick
        self._end_tick = len(self.prices) - 1
        self._last_trade_tick = self._current_tick - 1
        self._position = np.zeros(self.assets)
        self._position_history = (self.window_size * [None]) + [self._position]
        self.margin = self.initial_amount
        self.portfolio = [0]*self.assets
        self.PV = 0
        self.reserve = self.initial_amount
        self._total_reward = 0.
        self._total_profit = 1.
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def process_data(self):
        """
        Environment에서 반드시 불러야 하는 메소드. 데이터들을 맞는 윈도우 사이즈(frame_bound)로 불러와 Environment에 띄우기 위함.
        데이터들은 앞서 설정된
        :return:
        """
        signal_features = []
        for idx in range(self.assets):






