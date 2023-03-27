import gym
import torch as t
from torch import nn
from gat_attention import CapGATattentionGRU
from stable_baselines3.common.policies import ActorCriticPolicy


class Transpose(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class GATNetwork(nn.Module):
    """
    Custom network for policy and value function
    It receives as input the feature extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the feature_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """
    def __init__(self, feature_dim, timesteps, last_layer_dim_pi, last_layer_dim_vf):
        super(GATNetwork, self).__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.policy_net = nn.Sequential(CapGATattentionGRU(last_layer_dim_pi, timesteps, feature_dim),
                                        nn.Linear(last_layer_dim_pi, 1),
                                        Transpose(),
                                        nn.ReLU())
        self.value_net = nn.Sequential(CapGATattentionGRU(last_layer_dim_vf, timesteps, feature_dim),
                                       nn.Linear(last_layer_dim_pi, 1),
                                       Transpose(),
                                       nn.ReLU())

    def forward(self, features):
        """
        :return: (Tensor, Tensor) latent_policy, latent_value of the specified network.
            if all layers are shared, the 'latent_policy == latent_value'
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class GATActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch, activation_fn, *args, **kwargs):
        super(GATActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch,
                                                   activation_fn, *args, **kwargs)
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GATNetwork(last_layer_dim_pi=self.action_space.shape[0],
                                        last_layer_dim_vf=self.action_space.shape[0],
                                        timesteps=self.observation_space.shape[1],
                                        feature_dim=self.observation_space.shape[2])





















