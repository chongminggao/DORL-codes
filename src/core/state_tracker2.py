# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 9:49 上午
# @Author  : Chongming GAO
# @FileName: state_tracker.py
import math

import numpy as np
import torch

from core.inputs import SparseFeatP, input_from_feature_columns, create_embedding_matrix
from deepctr_torch.inputs import varlen_embedding_lookup, get_varlen_pooling_list, \
    VarLenSparseFeat, DenseFeat, combined_dnn_input

from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from core.layers import PositionalEncoding
from core.user_model import build_input_features, compute_input_dim

FLOAT = torch.FloatTensor


class StateTrackerBase2(nn.Module):
    def __init__(self, user_columns, action_columns, feedback_columns,
                 have_user_embedding=True, have_action_embedding=True, have_feedback_embedding=False,
                 use_pretrained_embedding=False, saved_embedding=None,
                 dataset="VirtualTB-v0",
                 device='cpu', seed=2021,
                 init_std=0.0001, MAX_TURN=100):
        super(StateTrackerBase2, self).__init__()
        torch.manual_seed(seed)

        self.dataset = dataset

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
            self.embedding_dict = saved_embedding.to(device)
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


class StateTrackerAvg2(nn.Module):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, saved_embedding, device,
                 use_userEmbedding=False, window=10, MAX_TURN=100):
        super(StateTrackerAvg2, self).__init__()
        self.user_columns = user_columns
        self.action_columns = action_columns
        self.feedback_columns = feedback_columns

        self.user_index = build_input_features(user_columns)
        self.action_index = build_input_features(action_columns)
        self.feedback_index = build_input_features(feedback_columns)

        self.dim_model = dim_model
        assert saved_embedding is not None
        self.embedding_dict = saved_embedding.to(device)

        self.dim_item = self.embedding_dict.feat_item.weight.shape[1]

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

        # X[-1][0] = np.nan
        # ind_nan = np.isnan(X)
        ind_nan = X == -1

        if any(ind_nan):
            X_normal = np.expand_dims(X[~ind_nan], -1)
            dim_res = sum([column.embedding_dim for column in feat_columns])

            X_res = torch.zeros([len(X), dim_res], device=self.device)
            nn.init.normal_(X_res, mean=0, std=0.001)

            # self.embedding_dict.feat_item.weight.requires_grad
            sparse_embedding_list, dense_value_list = input_from_feature_columns(FLOAT(X_normal).to(self.device),
                                                                                 feat_columns,
                                                                                 self.embedding_dict, feat_index,
                                                                                 support_dense=True, device=self.device)
            new_X = combined_dnn_input(sparse_embedding_list, dense_value_list)

            X_res[~ind_nan.squeeze()] = new_X
        else:
            X_normal = X
            sparse_embedding_list, dense_value_list = input_from_feature_columns(FLOAT(X_normal).to(self.device),
                                                                                 feat_columns,
                                                                                 self.embedding_dict, feat_index,
                                                                                 support_dense=True, device=self.device)
            new_X = combined_dnn_input(sparse_embedding_list, dense_value_list)
            X_res = new_X

        return X_res

    def forward(self, buffer=None, indices=None, obs=None,
                reset=None, is_obs=None):

        if reset:  # get user embedding

            users = np.expand_dims(obs[:, 0], -1)
            items = np.expand_dims(obs[:, 1], -1)

            # e_i = torch.ones(obs.shape[0], self.dim_item, device=self.device)
            # nn.init.normal_(e_i, mean=0, std=0.0001)

            e_i = self.get_embedding(items, "action")

            if self.use_userEmbedding:
                e_u = self.get_embedding(users, "user")
                # s0 = self.ffn_user(e_u)
                s0 = torch.cat([e_u, e_i], dim=-1)
            else:
                s0 = e_i

            return s0

        else:
            index = indices
            flag_has_init = np.zeros_like(index, dtype=bool)
            #
            # if is_obs:
            #     live_id_now = buffer.prev(index) != index
            #     index = buffer.prev(index)
            # else:
            #     live_id_now = np.ones_like(index, dtype=bool)

            # ind_init = ~live_id_now & ~flag_has_init
            # obs_all = buffer[index].obs_next
            # rew_all = buffer[index].rew
            obs_all = np.zeros([0, 2], dtype=int)
            rew_all = np.zeros([0])


            live_mat = np.zeros([0, len(index)], dtype=bool)

            first_flag = True

            '''
            Logic: Always use obs_next(t) and reward(t) to construct state(t+1), since obs_next(t) == obs(t+1).
            Note: The inital obs(0) == obs_next(-1) and reward(-1) are not recorded. So we have to initialize them.  
            '''
            while not all(flag_has_init) and len(live_mat) < self.window:
                if is_obs or not first_flag:
                    live_id_prev = buffer.prev(index) != index
                    index = buffer.prev(index)
                else:
                    live_id_prev = np.ones_like(index, dtype=bool)

                first_flag = False
                # live_id_prev = buffer.prev(index) != index

                ind_init = ~live_id_prev & ~flag_has_init
                obs_prev = buffer[index].obs_next
                rew_prev = buffer[index].rew

                obs_prev[ind_init, 1] = -1 # initialize obs
                rew_prev[ind_init] = 1     # initialize reward.
                flag_has_init[ind_init] = True
                live_id_prev[ind_init] = True

                live_mat = np.vstack([live_mat, live_id_prev])

                obs_all = np.concatenate([obs_all, obs_prev])
                rew_all = np.concatenate([rew_all, rew_prev])

            user_all = np.expand_dims(obs_all[:, 0], -1)
            item_all = np.expand_dims(obs_all[:, 1], -1)

            e_i = self.get_embedding(item_all, "action")

            rew_matrix = rew_all.reshape((-1, 1))
            e_r = self.get_embedding(rew_matrix, "feedback")

            if self.use_userEmbedding:
                e_u = self.get_embedding(user_all, "user")
                s_t = torch.cat([e_u, e_i], dim=-1)
            else:
                s_t = e_i

            state_flat = s_t * e_r
            state_cube = state_flat.reshape((-1, len(index), state_flat.shape[-1]))

            mask = torch.from_numpy(np.expand_dims(live_mat, -1)).to(self.device)
            state_masked = state_cube * mask

            state_sum = state_masked.sum(dim=0)
            state_final = state_sum / torch.from_numpy(np.expand_dims(live_mat.sum(0), -1)).to(self.device)

            # BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

            # index = indices
            # flag_has_init = np.zeros_like(index, dtype=bool)
            #
            # if is_obs:
            #     live_id_now = buffer.prev(index) != index
            #     index = buffer.prev(index)
            # else:
            #     live_id_now = np.ones_like(index, dtype=bool)
            #
            # ind_init = ~live_id_now & ~flag_has_init
            # obs_all = buffer[index].obs_next
            # rew_all = buffer[index].rew
            #
            # obs_all[ind_init, 1] = -1
            # rew_all[ind_init] = 1
            # flag_has_init[ind_init] = True
            # live_id_now[ind_init] = True
            #
            # live_mat2 = np.expand_dims(live_id_now,0)
            # live_id_prev = buffer.prev(index) != index
            #
            #
            # while not all(flag_has_init):
            #
            #     index = buffer.prev(index)
            #
            #     ind_init = ~live_id_prev & ~flag_has_init
            #     obs_prev = buffer[index].obs_next
            #     rew_prev = buffer[index].rew
            #
            #     obs_prev[ind_init, 1] = -1
            #     rew_prev[ind_init] = 1
            #     flag_has_init[ind_init] = True
            #     live_id_prev[ind_init] = True
            #
            #     live_mat2 = np.vstack([live_mat2, live_id_prev])
            #
            #     obs_all = np.concatenate([obs_all, obs_prev])
            #     rew_all = np.concatenate([rew_all, rew_prev])
            #
            #     live_id_prev = buffer.prev(index) != index
            #
            # user_all2 = np.expand_dims(obs_all[:, 0], -1)
            # item_all2 = np.expand_dims(obs_all[:, 1], -1)
            #
            # e_i = self.get_embedding(item_all2, "action")
            #
            # rew_matrix = rew_all.reshape((-1, 1))
            # e_r = self.get_embedding(rew_matrix, "feedback")
            #
            # if self.use_userEmbedding:
            #     e_u = self.get_embedding(user_all2, "user")
            #     s_t = torch.cat([e_u, e_i], dim=-1)
            # else:
            #     s_t = e_i
            #
            # state_flat = s_t * e_r
            # state_cube = state_flat.reshape((-1, len(index), state_flat.shape[-1]))
            #
            # mask = np.expand_dims(live_mat2, -1)
            # state_masked = state_cube * mask
            #
            # state_sum = state_masked.sum(dim=0)
            # state_final2 = state_sum / np.expand_dims(live_mat2.sum(0), -1)
            #
            # assert all(item_all2 == item_all)
            # assert all(user_all2 == user_all)
            # assert all(live_mat2.reshape(-1) == live_mat.reshape(-1))

            return state_final

    def build_state(self, obs=None,
                    env_id=None,
                    obs_next=None,
                    reset=False, **kwargs):
        if reset:
            self.user = None
            return

        if obs is not None:  # 1. initialize the state vectors
            self.user = obs
            # item = np.ones_like(obs) * np.nan
            item = np.ones_like(obs) * -1
            ui_pair = np.hstack([self.user, item])
            res = {"obs": ui_pair}

        elif obs_next is not None:  # 2. add action autoregressively
            item = obs_next
            user = self.user[env_id]
            ui_pair = np.hstack([user, item])
            res = {"obs_next": ui_pair}

        return res


class StateTrackerTransformer2(StateTrackerBase2):
    def __init__(self, user_columns, action_columns, feedback_columns,
                 dim_model, dim_state, dim_max_batch, dropout=0.1,
                 dataset="VirtualTB-v0",
                 have_user_embedding=True, have_action_embedding=True, have_feedback_embedding=False,
                 use_pretrained_embedding=False, saved_embedding=None,
                 nhead=8, d_hid=128, nlayers=2,
                 device='cpu', seed=2021,
                 init_std=0.0001, padding_idx=None, MAX_TURN=100):

        super(StateTrackerTransformer2, self).__init__(user_columns, action_columns, feedback_columns,
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
            else:  # "KuaiEnv-v0":
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
            else:  # "KuaiEnv-v0":
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


class StateTracker_Caser(nn.Module):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, device,
                 use_userEmbedding=False, window_size=10, filter_sizes=[2, 3, 4], num_filters=16,
                 dropout_rate=0.1):
        super(StateTracker_Caser, self).__init__()
        self.user_columns = user_columns
        self.action_columns = action_columns
        self.feedback_columns = feedback_columns

        self.user_index = build_input_features(user_columns)
        self.action_index = build_input_features(action_columns)
        self.feedback_index = build_input_features(feedback_columns)

        self.dim_model = dim_model
        self.window_size = window_size
        self.device = device
        self.use_userEmbedding = use_userEmbedding
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        self.hidden_size = action_columns[0].embedding_dim
        self.num_item = action_columns[0].vocabulary_size

        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.final_dim = self.hidden_size + self.num_filters_total

        # Item embedding
        embedding_dict = torch.nn.ModuleDict(
            {"feat_item": torch.nn.Embedding(num_embeddings=self.num_item + 1,
                                             embedding_dim=self.hidden_size)})
        self.embedding_dict = embedding_dict.to(device)
        nn.init.normal_(self.embedding_dict.feat_item.weight, mean=0, std=0.1)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.window_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def build_state(self, obs=None,
                    env_id=None,
                    obs_next=None,
                    reset=False, **kwargs):
        if reset:
            self.user = None
            return

        if obs is not None:  # 1. initialize the state vectors
            self.user = obs
            # item = np.ones_like(obs) * np.nan
            item = np.ones_like(obs) * self.num_item
            ui_pair = np.hstack([self.user, item])
            res = {"obs": ui_pair}

        elif obs_next is not None:  # 2. add action autoregressively
            item = obs_next
            user = self.user[env_id]
            ui_pair = np.hstack([user, item])
            res = {"obs_next": ui_pair}

        return res

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

        # X[-1][0] = np.nan
        # ind_nan = np.isnan(X)

        X[X == -1] = self.num_item

        sparse_embedding_list, dense_value_list = input_from_feature_columns(FLOAT(X).to(self.device), feat_columns,
                                                                             self.embedding_dict, feat_index,
                                                                             support_dense=True, device=self.device)
        new_X = combined_dnn_input(sparse_embedding_list, dense_value_list)
        X_res = new_X

        return X_res

    def convert_to_k_state_embedding(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None):
        if reset:  # get user embedding
            # users = np.expand_dims(obs[:, 0], -1)
            items = np.expand_dims(obs[:, 1], -1)
            items_window = items.repeat(self.window_size, axis=1)

            # a = self.get_embedding(items_window, "action")

            e_i = self.get_embedding(items, "action")
            emb_state = e_i.repeat_interleave(self.window_size, dim=0).reshape([len(e_i), self.window_size, -1])

            return emb_state, items_window

        else:
            index = indices
            flag_has_init = np.zeros_like(index, dtype=bool)

            obs_all = np.zeros([0, 2], dtype=int)
            rew_all = np.zeros([0])
            live_mat = np.zeros([0, len(index)], dtype=bool)

            first_flag = True
            '''
                Logic: Always use obs_next(t) and reward(t) to construct state(t+1), since obs_next(t) == obs(t+1).
                Note: The inital obs(0) == obs_next(-1) and reward(-1) are not recorded. So we have to initialize them.  
            '''
            # while not all(flag_has_init) and len(live_mat) < self.window_size:
            while len(live_mat) < self.window_size:
                if is_obs or not first_flag:
                    live_id_prev = buffer.prev(index) != index
                    index = buffer.prev(index)
                else:
                    live_id_prev = np.ones_like(index, dtype=bool)

                first_flag = False
                # live_id_prev = buffer.prev(index) != index

                ind_init = ~live_id_prev & ~flag_has_init # just dead and have not been initialized before.
                obs_prev = buffer[index].obs_next
                rew_prev = buffer[index].rew

                obs_prev[~live_id_prev, 1] = self.num_item
                rew_prev[~live_id_prev] = 1
                # obs_prev[ind_init, 1] = self.num_item
                # rew_prev[ind_init] = 1
                flag_has_init[ind_init] = True
                live_id_prev[ind_init] = True

                live_mat = np.vstack([live_mat, live_id_prev])

                obs_all = np.concatenate([obs_all, obs_prev])
                rew_all = np.concatenate([rew_all, rew_prev])

            user_all = np.expand_dims(obs_all[:, 0], -1)
            item_all = np.expand_dims(obs_all[:, 1], -1)

            e_i = self.get_embedding(item_all, "action")

            # rew_matrix = rew_all.reshape((-1, 1))
            # e_r = self.get_embedding(rew_matrix, "feedback")

            # if self.use_userEmbedding:
            #     e_u = self.get_embedding(user_all, "user")
            #     s_t = torch.cat([e_u, e_i], dim=-1)
            # else:
            s_t = e_i

            # state_flat = s_t * e_r
            state_flat = s_t
            state_cube = state_flat.reshape((-1, len(index), state_flat.shape[-1]))

            mask = torch.from_numpy(np.expand_dims(live_mat, -1)).to(self.device)
            state_masked = state_cube * mask

            state_sum = state_masked.sum(dim=0)
            state_final = state_sum / torch.from_numpy(np.expand_dims(live_mat.sum(0), -1)).to(self.device)

            return state_final

    def forward(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None):

        emb_state, items_window = self.convert_to_k_state_embedding(buffer, indices, obs, reset, is_obs)

        mask = torch.ne(FLOAT(items_window), self.num_item).unsqueeze(-1)
        emb_state_masked = emb_state * mask
        emb_state_final = emb_state_masked.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(emb_state_final))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(emb_state_final))
        v_flat = v_out.view(-1, self.hidden_size)

        state_hidden = torch.cat([h_pool_flat, v_flat], 1)
        state_hidden_dropout = self.dropout(state_hidden)

        return state_hidden_dropout
