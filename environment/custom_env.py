import gym
import numpy as np

from numpy import ndarray
from sklearn.preprocessing import StandardScaler


# noinspection PyTypeChecker
class MultiStockTradingEnv(gym.Env):
    """

    """
    metadata = {"render.modes": ["humans"]}

    def __init__(self, dfs, price_df, initial_amount, trade_cost, num_features, num_stocks,
                 window_size, frame_bound, scalers=None, tech_indicator_list=[], reward_scaling=1e-5,
                 representative=None, supresention_rate=0.66):
        """
        :param dfs: 기업별 가상화폐별 raw 데이터를 담고있는 리스트
        :param price_df: 기업별 가상화폐별 종가를 담고있는 리스트
        :param initial_amount: 모델 training 시 사용될 초기 금액
        :param trade_cost: 거래비용(주식 사고팔때 발생하는 그것)
        :param num_features: 각 기업별 사용될 feature 들
        :param num_stocks: 트레이딩에 사용될 기업들 숫자
        :param window_size: 다음 Action을 할 때 고려될 이전 시간
        :param frame_bound: training/test에 사용될 start index, end index 범위. window size보단 커야함
        :param scalers: 각각의 주식을 scaling 할 scaler 리스트
        :param tech_indicator_list: 주식 기술지표 리스트
        :param reward_scaling: 각 주식 reward의 scaling 값
        :param representative:
        :param supresention_rate:
        """
        if len(tech_indicator_list) != 0:
            num_features = len(tech_indicator_list)
        self.dfs = dfs
        self.price_df = price_df
        self.initial_amount = initial_amount
        self.trade_cost = trade_cost
        self.state_space = num_features
        self.assets = num_stocks
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.scalers = scalers
        self.tech_indicators = tech_indicator_list
        self.reward_scaling = reward_scaling
        self.margin = initial_amount
        self.portfolio = [0] * num_stocks
        self.portfolio_value = 0
        self.reserve = initial_amount

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.price_df) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = np.zeros(self.assets)
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = []
        self.rewards = []
        self.pvs = []

        if scalers is None:
            self.scalers = [None] * self.assets
        else:
            self.scalers = scalers

        self.representative = representative
        self.suppression_rate = supresention_rate

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
        self.portfolio_value = 0
        self.reserve = self.initial_amount
        self._total_reward = 0.
        self._total_profit = 1.
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def process_data(self):
        """
        Environment에서 반드시 불러야 하는 메소드. 데이터들을 맞는 윈도우 사이즈(frame_bound)로 불러와 Environment에 띄우기 위함.
        데이터들은 이 method에서 scaling됨.
        :return: window size로 자른 종가, 정규화되어 윈도우 사이즈로 잘린 기술적 지표
        """
        signal_features = []
        for idx in range(self.assets):
            df = self.dfs[idx]
            start = self.frame_bound[0] - self.window_size
            end = self.frame_bound[1]

            if self.scalers[idx]:
                current_scaler = self.scalers[idx]
                signal_features_i = current_scaler.transform(df.loc[:, self.tech_indicators])[start:end]
            else:
                current_scaler = StandardScaler()
                signal_features_i = current_scaler.fit_transform(df.loc[:, self.tech_indicators])[start:end]
                signal_features[idx] = current_scaler
            signal_features.append(signal_features_i)

        self.prices = self.price_df.loc[:, :].to_numpy()[start:end]

        if self.representative:
            self.representative = self.price_df.loc[:, self.representative].to_numpy()[start:end]
        else:
            self.representative = self.price_df.loc[:, 'SENSEX'].to_numpy()[start:end]

        self.signal_features = np.array(signal_features)
        self._end_tick = len(self.prices) - 1
        return self.prices, self.signal_features

    def _get_observation(self):
        pass

    def step(self, actions):
        self._done = False
        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._done = True

        current_prices = self.prices[self._current_tick]
        current_prices[np.isnan(current_prices)] = 0
        current_prices_for_division = current_prices
        current_prices_for_division[current_prices_for_division == 0] = 1e9

        abs_portfolio_dist = abs(actions)
        N = int(np.round(abs_portfolio_dist.size*0.66))
        abs_portfolio_dist[np.argpartition(abs_portfolio_dist, kth=N)[:N]] = 0

        self.margin = self.reserve + sum(self.portfolio * current_prices)

        norm_margin_pos = (abs_portfolio_dist/sum(abs_portfolio_dist)) * self.margin
        next_positions = np.sign(actions) * norm_margin_pos
        change_in_positions = next_positions - self._position
        actions_in_market = np.divide(change_in_positions, current_prices_for_division).astype(int)










