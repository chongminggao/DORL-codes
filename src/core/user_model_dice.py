# -*- coding: utf-8 -*-
import torch
from deepctr_torch.inputs import combined_dnn_input, build_input_features
from deepctr_torch.layers import DNN, PredictionLayer, FM
from torch import nn

from core.inputs import input_from_feature_columns
from core.layers import Linear, create_embedding_matrix
from core.user_model import UserModel, compute_input_dim


class UserModel_DICE(UserModel):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param device: str, ``"cpu"`` or ``"cuda:0"``

    :return: A PyTorch model instance.
    """

    def __init__(self, feature_columns, y_columns, task, task_logit_dim,
                 dnn_hidden_units=(128, 128),
                 l2_reg_embedding=1e-5, l2_reg_dnn=1e-1, init_std=0.0001, task_dnn_units=None, seed=2021, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, device='cpu', ab_columns=None):

        super(UserModel_DICE, self).__init__(feature_columns, y_columns,
                                             l2_reg_embedding=l2_reg_embedding,
                                             init_std=init_std, seed=seed, device=device)

        self.feature_columns = feature_columns
        self.feature_index = self.feature_index

        self.y_columns = y_columns
        self.task_logit_dim = task_logit_dim

        self.sigmoid = nn.Sigmoid()
        """
        For MMOE Layer
        """
        self.task = task
        self.task_dnn_units = task_dnn_units

        """
        Prepare DICE conformity/interest features
        """

        self.feature_main = self.feature_columns
        name2index = {f.name:i for i,f in enumerate(feature_columns)}
        DICE_index = [name2index[name] for name in ['user_id','item_id','user_id_int','item_id_int']]
        self.feature_ui_con = [self.feature_columns[DICE_index[0]]] + [self.feature_columns[DICE_index[1]]]
        self.feature_ui_int = [self.feature_columns[DICE_index[2]]] + [self.feature_columns[DICE_index[3]]]
        
        #print(self.feature_ui_con,self.feature_ui_con)

        # self.feature_index_main = OrderedDict({k: v for i, (k, v) in enumerate(self.feature_index.items()) if i < 9})
        # self.feature_index_ui = OrderedDict({k: v for i, (k, v) in enumerate(self.feature_index.items()) if i in [1, 3]})
        self.index_main = build_input_features(self.feature_main)
        self.index_ui_int = build_input_features(self.feature_ui_int)
        self.index_ui_con = build_input_features(self.feature_ui_con)

        """
        For DNN Layer
        """

        self.dnn_main = DNN(compute_input_dim(self.feature_main), dnn_hidden_units,
                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                            init_std=init_std, device=device)
        self.last_main = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.out_main = PredictionLayer(task, 1)

        self.dnn_ui = DNN(compute_input_dim(self.feature_ui_int), dnn_hidden_units,
                          activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                          init_std=init_std, device=device)
        self.last_ui = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.out_ui = PredictionLayer(task, 1)

        """
        For DeepFM Layer.
        """
        use_fm = True if task_logit_dim == 1 else False
        self.use_fm = use_fm

        self.fm_task = FM() if use_fm else None

        self.linear_main = Linear(self.feature_main, self.index_main, device=device)
        self.linear_ui = Linear(self.feature_ui_int, self.index_ui_int, device=device)

        self.add_regularization_weight(self.parameters(), l2=l2_reg_dnn)

        if ab_columns is not None:
            raise RuntimeError('ab columns is not supported by DICE!')

        self.to(device)
        

    def _deepfm(self, X, feature_columns, feature_index, score=None, is_main=True):

        sparse_embedding_list, dense_value_list = input_from_feature_columns(X, feature_columns, self.embedding_dict,
                                                                                  feature_index,
                                                                                  support_dense=True, device=self.device)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if is_main:
            linear_model = self.linear_main
            dnn = self.dnn_main
            last = self.last_main
            out = self.out_main
        else:
            linear_model = self.linear_ui
            dnn = self.dnn_ui
            last = self.last_ui
            out = self.out_ui

        # Linear and FM logit
        logit = torch.zeros([len(X), self.task_logit_dim], device=X.device)

        if linear_model is not None:
            logit = logit + linear_model(X)

            fm_model = self.fm_task
            if self.use_fm and len(sparse_embedding_list) > 0 and fm_model is not None:
                fm_input = torch.cat(sparse_embedding_list, dim=1)
                logit += fm_model(fm_input)

        linear_logit = logit

        # DNN
        dnn_logit = out(last(dnn(dnn_input)))

        y_pred = linear_logit + dnn_logit

        y_score = y_pred * score if score is not None else y_pred

        return y_score

    def get_loss(self, x, y, score):

        assert x.shape[1] % 2 == 0
        num_features = x.shape[1] // 2
        X_pos = x[:, :num_features]
        X_neg = x[:, num_features:]


        y_deepfm_pos = self._deepfm(X_pos, self.feature_main, self.index_main, is_main=True)
        y_deepfm_neg = self._deepfm(X_neg, self.feature_main, self.index_main, is_main=True)

        ucon,icon,uint,iint = [self.feature_index[name][0] for name in ['user_id','item_id','user_id_int','item_id_int']]

        X_pos_con = x[:,[ucon,icon]]
        X_pos_int = x[:,[uint,iint]]
        X_neg_con = x[:,[num_features+ucon,num_features+icon]]
        X_neg_int = x[:,[num_features+uint,num_features+iint]]

        y_deepfm_pos_int = self._deepfm(X_pos_int, self.feature_ui_int, self.index_ui_int, is_main=False)
        y_deepfm_neg_int = self._deepfm(X_neg_int, self.feature_ui_int, self.index_ui_int, is_main=False)
        y_deepfm_pos_con = self._deepfm(X_pos_con, self.feature_ui_con, self.index_ui_con, is_main=False)
        y_deepfm_neg_con = self._deepfm(X_neg_con, self.feature_ui_con, self.index_ui_con, is_main=False)

        loss = self.loss_func(y, y_deepfm_pos, y_deepfm_neg,
                              y_deepfm_pos_int, y_deepfm_neg_int,
                              y_deepfm_pos_con, y_deepfm_neg_con, score)

        return loss

    def forward(self, x, score=None):
        ucon,icon,uint,iint = [self.feature_index[name][0] for name in ['user_id','item_id','user_id_int','item_id_int']]
        x2 = torch.concat([x,x[:,[ucon,icon]]],dim=1)
        assert x2.shape[1] - 1 == iint
        assert x2.shape[1] - 2 == uint       
        y_deepfm = self._deepfm(x2, self.feature_main, self.index_main, score=score, is_main=True)
        return y_deepfm
