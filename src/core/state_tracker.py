# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 9:49 上午
# @Author  : Chongming GAO
# @FileName: state_tracker.py
import math

import torch

from core.inputs import SparseFeatP, input_from_feature_columns, create_embedding_matrix
from deepctr_torch.inputs import varlen_embedding_lookup, get_varlen_pooling_list, \
    VarLenSparseFeat, DenseFeat, combined_dnn_input

from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from core.layers import PositionalEncoding
from core.user_model import build_input_features, compute_input_dim

FLOAT = torch.FloatTensor


class StateTrackerBase(nn.Module):
    def __init__(self, user_columns, action_columns, feedback_columns,
                 have_user_embedding=True, have_action_embedding=True, have_feedback_embedding=False,
                 use_pretrained_embedding=False, saved_embedding=None,
                 dataset="VirtualTB-v0",
                 device='cpu', seed=2021,
                 init_std=0.0001, MAX_TURN=100):
        super(StateTrackerBase, self).__init__()
        torch.manual_seed(seed)

        self.dataset = dataset

        # self.user_index = build_input_features(user_columns)
        # self.action_index = build_input_features(action_columns)

        # self.reg_loss = torch.zeros((1,), device=device)
        # self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.MAX_TURN = MAX_TURN

        self.have_user_embedding = have_user_embedding
        self.have_action_embedding = have_action_embedding
        self.have_feedback_embedding = have_feedback_embedding

        self.user_columns = user_columns
        self.action_columns = action_columns
        self.feedback_columns = feedback_columns

        all_columns = []
        if not have_user_embedding:
            all_columns += user_columns
            self.user_index = build_input_features(user_columns)
        if not have_action_embedding:
            all_columns += action_columns
            self.action_index = build_input_features(action_columns)
        if not have_feedback_embedding:
            all_columns += feedback_columns
            self.feedback_index = build_input_features(feedback_columns)

        if use_pretrained_embedding:
            assert saved_embedding is not None
            self.embedding_dict = saved_embedding
        else:
            self.embedding_dict = create_embedding_matrix(all_columns, init_std, sparse=False, device=device)

    def get_embedding(self, X, type):
        if type == "user":
            have_embedding = self.have_user_embedding
        elif type == "action":
            have_embedding = self.have_action_embedding
        elif type == "feedback":
            have_embedding = self.have_feedback_embedding
        if have_embedding:
            return FLOAT(X).to(self.device)

        if type == "user":
            feat_columns = self.user_columns
            feat_index = self.user_index
        elif type == "action":
            feat_columns = self.action_columns
            feat_index = self.action_index
        elif type == "feedback":
            feat_columns = self.feedback_columns
            feat_index = self.feedback_index

        # sparse_embedding_list, dense_value_list = \
        #     input_from_feature_columns(FLOAT(X).to(self.device), feat_columns, self.embedding_dict, feat_index,
        #                                self.device)

        sparse_embedding_list, dense_value_list = input_from_feature_columns(FLOAT(X).to(self.device), feat_columns,
                                                                             self.embedding_dict, feat_index,
                                                                             support_dense=True, device=self.device)

        new_X = combined_dnn_input(sparse_embedding_list, dense_value_list)

        return new_X

    def build_state(self,
                    obs=None,
                    env_id=None,
                    obs_next=None,
                    rew=None,
                    done=None,
                    info=None,
                    policy=None):
        return {}


class StateTrackerAvg(nn.Module):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, saved_embedding, device,
                 use_userEmbedding=False, window=10, MAX_TURN=100):
        super(StateTrackerAvg, self).__init__()
        self.user_columns = user_columns
        self.action_columns = action_columns
        self.feedback_columns = feedback_columns

        self.user_index = build_input_features(user_columns)
        self.action_index = build_input_features(action_columns)
        self.feedback_index = build_input_features(feedback_columns)

        self.dim_model = dim_model
        assert saved_embedding is not None
        self.embedding_dict = saved_embedding

        self.window = window



        self.device = device
        self.MAX_TURN = MAX_TURN

        self.use_userEmbedding = use_userEmbedding
        if self.use_userEmbedding:
            self.ffn_user = nn.Linear(compute_input_dim(self.user_columns), self.dim_model, device=self.device)

    def get_embedding(self, X, type):
        if type == "user":
            feat_columns = self.user_columns
            feat_index = self.user_index
        elif type == "action":
            feat_columns = self.action_columns
            feat_index = self.action_index
        elif type == "feedback":
            feat_columns = self.feedback_columns
            feat_index = self.feedback_index

        sparse_embedding_list, dense_value_list = input_from_feature_columns(FLOAT(X).to(self.device), feat_columns,
                                                                             self.embedding_dict, feat_index,
                                                                             support_dense=True, device=self.device)
        new_X = combined_dnn_input(sparse_embedding_list, dense_value_list)

        return new_X

    def build_state(self, obs=None,
                    env_id=None,
                    obs_next=None,
                    rew=None,
                    done=None,
                    info=None,
                    policy=None,
                    dim_batch=None,
                    reset=False):
        if reset and dim_batch:
            self.data = torch.zeros(self.MAX_TURN, dim_batch, self.dim_model,
                                    device=self.device)  # (Length, Batch, Dim)
            self.len_data = torch.zeros(dim_batch, dtype=torch.int64)
            return
        res = {}

        if obs is not None:  # 1. initialize the state vectors
            if self.use_userEmbedding:
                e_u = self.get_embedding(obs, "user")
                s0 = self.ffn_user(e_u)
            else:
                s0 = torch.ones(obs.shape[0], self.dim_model)
                s0 = nn.init.normal_(s0, mean=0, std=0.0001)

            self.data[0, env_id, :] = s0
            self.len_data[env_id] = 1
            res = {"obs": s0}

        elif obs_next is not None:  # 2. add action autoregressively

            a_t = self.get_embedding(obs_next, "action")
            self.len_data[env_id] += 1
            length = int(self.len_data[env_id[0]])

            # turn = obs_next[:, -1]
            # assert all(self.len_data[env_id].numpy() == turn + 1)
            rew_matrix = rew.reshape((-1, 1))
            r_t = self.get_embedding(rew_matrix, "feedback")

            self.data[length - 1, env_id, :] = a_t * r_t
            if length <= self.window:
                s_t = self.data[:length, env_id].mean(dim=0)
            else:
                if self.use_userEmbedding:
                    self.data[length - self.window, env_id] = self.data[0, env_id] # Copy operation!
                s_t = self.data[length-self.window:length, env_id].mean(dim=0)

            res = {"obs_next": s_t}

        return res
        # return {"obs": obs, "env_id": env_id, "obs_next": obs_next, "rew": rew,
        #         "done": done, "info": info, "policy": policy}


class StateTrackerTransformer(StateTrackerBase):
    def __init__(self, user_columns, action_columns, feedback_columns,
                 dim_model, dim_state, dim_max_batch, dropout=0.1,
                 dataset="VirtualTB-v0",
                 have_user_embedding=True, have_action_embedding=True, have_feedback_embedding=False,
                 use_pretrained_embedding=False, saved_embedding=None,
                 nhead=8, d_hid=128, nlayers=2,
                 device='cpu', seed=2021,
                 init_std=0.0001, padding_idx=None, MAX_TURN=100):

        super(StateTrackerTransformer, self).__init__(user_columns, action_columns, feedback_columns,
                                                      have_user_embedding=have_user_embedding,
                                                      have_action_embedding=have_action_embedding,
                                                      have_feedback_embedding=have_feedback_embedding,
                                                      use_pretrained_embedding=use_pretrained_embedding,
                                                      saved_embedding=saved_embedding,
                                                      dataset=dataset,
                                                      device=device, seed=seed, init_std=init_std, MAX_TURN=MAX_TURN)
        self.dim_model = dim_model
        self.ffn_user = nn.Linear(compute_input_dim(user_columns),
                                  dim_model, device=device)
        # self.fnn_gate = nn.Linear(3 * compute_input_dim(action_columns),
        #                           dim_model, device=device)
        self.fnn_gate = nn.Linear(1 + compute_input_dim(action_columns),
                                  compute_input_dim(action_columns), device=device)
        self.ffn_action = nn.Linear(compute_input_dim(action_columns),
                                    dim_model, device=device)

        self.sigmoid = nn.Sigmoid()

        self.pos_encoder = PositionalEncoding(dim_model, dropout)
        encoder_layers = TransformerEncoderLayer(dim_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = nn.Linear(dim_model, dim_state)
        self.dim_state = dim_state

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1

        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src0: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.dim_model)
        src = src0 * math.sqrt(self.dim_model)  # Added by Chongming
        src_p = self.pos_encoder(src)
        output = self.transformer_encoder(src_p, src_mask)
        output_t = output[-1, :, :]

        s_t = self.decoder(output_t)
        return s_t

    def build_state(self, obs=None,
                    env_id=None,
                    obs_next=None,
                    rew=None,
                    done=None,
                    info=None,
                    policy=None,
                    dim_batch=None,
                    reset=False):
        if reset and dim_batch:
            self.data = torch.zeros(self.MAX_TURN, dim_batch, self.dim_model,
                                    device=self.device)  # (Length, Batch, Dim)
            self.len_data = torch.zeros(dim_batch, dtype=torch.int64)
            return

        res = {}

        if obs is not None:  # 1. initialize the state vectors
            if self.dataset == "VirtualTB-v0":
                e_u = self.get_embedding(obs[:, :-3], "user")
            elif self.dataset == "KuaiEnv-v0":
                e_u = self.get_embedding(obs, "user")

            e_u_prime = self.ffn_user(e_u)

            length = 1
            self.len_data[env_id] = length
            self.data[0, env_id, :] = e_u_prime

            nowdata = self.data[:length, env_id, :]
            mask = torch.triu(torch.ones(length, length, device=self.device) * float('-inf'), diagonal=1)
            s0 = self.forward(nowdata, mask)

            res = {"obs": s0}


        elif obs_next is not None:  # 2. add action autoregressively
            if self.dataset == "VirtualTB-v0":
                a_t = self.get_embedding(obs_next[:, :-3], "action")
            elif self.dataset == "KuaiEnv-v0":
                a_t = self.get_embedding(obs_next, "action")

            self.len_data[env_id] += 1
            length = int(self.len_data[env_id[0]])

            # turn = obs_next[:, -1]
            # assert all(self.len_data[env_id].numpy() == turn + 1)
            rew_matrix = rew.reshape((-1, 1))
            r_t = self.get_embedding(rew_matrix, "feedback")

            # g_t = self.sigmoid(self.fnn_gate(torch.cat((r_t, a_t, r_t * a_t), -1)))
            g_t = self.sigmoid(self.fnn_gate(torch.cat((r_t, a_t), -1)))
            a_t_prime = g_t * a_t
            a_t_prime_reg = self.ffn_action(a_t_prime)

            self.data[length - 1, env_id, :] = a_t_prime_reg
            mask = torch.triu(torch.ones(length, length, device=self.device) * float('-inf'), diagonal=1)
            mask = mask

            s_t = self.forward(self.data[:length, env_id, :], mask)

            res = {"obs_next": s_t}

        return res
        # return {"obs": obs, "env_id": env_id, "obs_next": obs_next, "rew": rew,
        #         "done": done, "info": info, "policy": policy}


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)
        if pe[:, 0, 1::2].shape[-1] % 2 == 0:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, :-1]

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class StateTrackerGRU(StateTrackerBase):
    def __init__(self):
        pass


class StateTrackerLSTM(StateTrackerBase):
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
