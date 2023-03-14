import gym
from torch import nn
from typing import Callable
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
    """
    Policy 와 Value function 을 위한 Custom network
    feature extractor 를 통과한 결과가 input 으로 들어옴
    """
    def __init__(self, feature_dim: int, timesteps: int = 12, last_layer_dim_pi: int = 64, last_layer_dim_vf: int = 64):
        super(CustomNetwork, self).__init__()
        """
        :param feature_dim: feature extractor 를 통과한 feature 의 차원
        :param last_layer_dim_pi: policy network 의 last layer 에 있는 node 개수
        :param last_layer_dim_vf: value network 의 last layer 에 있는 node 개수
        """
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.policy_net = nn.Sequential(nn.Linear(last_layer_dim_pi * timesteps * feature_dim,
                                                  last_layer_dim_pi * feature_dim),
                                        nn.ReLU(),
                                        nn.Linear(last_layer_dim_pi * feature_dim, last_layer_dim_pi),
                                        nn.Tanh())
        self.value_net = nn.Sequential(nn.Linear(last_layer_dim_pi * timesteps * feature_dim,
                                                 last_layer_dim_pi * feature_dim),
                                       nn.ReLU(),
                                       nn.Linear(last_layer_dim_pi * feature_dim, last_layer_dim_pi),
                                       nn.Tanh())

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 lr_schedule: Callable[[float], float], net_arch=None, activation_fn=nn.Tanh, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(observation_space, action_space,
                                                      lr_schedule, net_arch, activation_fn, *args, **kwargs)

        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(last_layer_dim_pi=self.action_space.shape[0],
                                           last_layer_dim_vf=self.action_space.shape[0],
                                           timesteps=self.observation_space.shape[1],
                                           feature_dim=self.observation_space.shape[2])















