import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv
from config import config

config = config[os.environ["ENV"]]


class AttentionBlock(nn.Module):
    def __init__(self, time_step, dim):
        super(AttentionBlock, self).__init__()
        self.time_step = time_step
        self.attention_matrix = nn.Linear(time_step, time_step)

    def forward(self, inputs):
        inputs_t = torch.transpose(inputs, 2, 1) # batch_size, input_dim, time_step
        attention_weight = self.attention_matrix(inputs_t)
        attention_probs = F.softmax(attention_weight, dim=1)
        attention_probs = torch.transpose(attention_probs, 2, 1)
        attention_vec = torch.mul(attention_probs, inputs)
        attention_vec = torch.sum(attention_vec, dim=1)
        return attention_vec, attention_probs


class CapGATattentionGRU(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim, use_gru=True):
        super(CapGATattentionGRU, self).__init__()
        outer_edge = np.ones(shape=(2, input_dim**2))
        count = 0
        for idx in range(input_dim):
            for jdx in range(input_dim):
                outer_edge[0][count] = idx
                outer_edge[1][count] = jdx
                count += 1

        self.dim = hidden_dim
        self.input_dim = input_dim
        self.time_step = time_step
        self.outer_edge = outer_edge
        self.batch = 1

        self.inner_edge = torch.tensor(outer_edge, dtype=torch.int64).to(config.DEVICE)
        self.use_gru = use_gru

        if self.use_gru:
            self.temporal_encoder = nn.GRU(input_dim * hidden_dim, input_dim * hidden_dim, num_layers=2,
                                           bidirectional=False)

        self.inner_gat0 = GATv2Conv(hidden_dim, hidden_dim)
        self.inner_gat1 = GATv2Conv(hidden_dim, hidden_dim)
        self.attention = AttentionBlock(12, hidden_dim)

