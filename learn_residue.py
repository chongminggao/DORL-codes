# -*- coding: utf-8 -*-
# @Author  : Chongming GAO
# @FileName: learn_residue
import argparse
import datetime
import functools
import json
import os
import pickle
import random
import sys
import time
import logzero
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from environments.KuaiRec.env.KuaiEnv import compute_exposure_effect_kuaiRec

sys.path.extend(["./src", "./src/DeepCTR-Torch"])
from core.evaluation.metrics import get_ranking_results
from core.inputs import SparseFeatP, input_from_feature_columns
from core.static_dataset import StaticDataset
from core.user_model_pairwise import UserModel_Pairwise
from core.util import negative_sampling
from deepctr_torch.inputs import DenseFeat, build_input_features, combined_dnn_input

from util.utils import create_dir


def get_args_all():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")

    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument("--bpr_weight", type=float, default=0.5)
    parser.add_argument('--neg_K', default=5, type=int)

    # recommendation related:
    # parser.add_argument('--not_softmax', action="store_false")
    parser.add_argument('--is_softmax', dest='is_softmax', action='store_true')
    parser.add_argument('--no_softmax', dest='is_softmax', action='store_false')
    parser.set_defaults(is_softmax=True)

    parser.add_argument('--rankingK', default=(20, 10, 5), type=int, nargs="+")
    parser.add_argument('--max_turn', default=30, type=int)

    parser.add_argument('--is_all_item_ranking', dest='is_all_item_ranking', action='store_true')
    parser.add_argument('--no_all_item_ranking', dest='is_all_item_ranking', action='store_false')
    parser.set_defaults(all_item_ranking=False)

    parser.add_argument('--l2_reg_dnn', default=0.1, type=float)
    parser.add_argument('--lambda_ab', default=10, type=float)

    parser.add_argument('--epsilon', default=0, type=float)
    parser.add_argument('--is_ucb', dest='is_ucb', action='store_true')
    parser.add_argument('--no_ucb', dest='is_ucb', action='store_false')
    parser.set_defaults(is_ucb=False)

    parser.add_argument("--feature_dim", type=int, default=8)
    parser.add_argument("--entity_dim", type=int, default=8)
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument('--dnn', default=(64, 64), type=int, nargs="+")
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--cuda', default=0, type=int)

    # exposure parameters:
    parser.add_argument('--tau', default=0, type=float)

    parser.add_argument('--is_ab', dest='is_ab', action='store_true')
    parser.add_argument('--no_ab', dest='is_ab', action='store_false')
    parser.set_defaults(is_ab=False)
    parser.add_argument("--message", type=str, default="point")

    args = parser.parse_known_args()[0]
    return args


def get_residue():



