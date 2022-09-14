# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd



def get_MRR(rec_list, true_list, true_rel):
    """ 平均倒排名 """
    rr = 0.
    for i in range(len(rec_list)):
        for j in range(len(rec_list[i])):
            # if true_list[i][0] == rec_list[i][j]:
            if rec_list[i][j] in true_list[i]:
                rr += 1/(j+1)  # 注意j的取值从0开始
                break
    mrr = rr / len(true_list)
    return mrr


def get_HR(rec_list, true_list, true_rel):
    """ 命中率 """
    hits = 0.
    for i in range(len(rec_list)):
        recom_set = set(rec_list[i])
        truth_set = set(true_list[i])
        n_union = len(recom_set & truth_set)
        if n_union > 0:
            hits += 1

    return hits / len(true_list)


def get_AP(rec_list, true_list, true_rel):
    """ 精度均值（Average Precision，简称AP) """
    hits = 0
    sum_precs = 0
    for i in range(len(rec_list)):
        if rec_list[i] in true_list:
            hits += 1
            sum_precs += hits / (i + 1.0)
    if hits > 0:
        return sum_precs / len(true_list)
    else:
        return 0


def get_MAP(rec_list, true_list, true_rel):
    """ 平均精度均值（Mean Average Precision，简称MAP) """
    ap = 0
    mAP = 0
    for i in range(len(rec_list)):
        ap += get_AP(rec_list[i], true_list[i])
    mAP = ap / len(true_list)
    return mAP


def get_Precision(rec_list, true_list, true_rel):
    """ 精确率 """
    n_union = 0.
    recommend_sum = 0.
    res = []
    for i in range(len(rec_list)):
        recom_set = set(rec_list[i])
        truth_set = set(true_list[i])
        hit = len(recom_set & truth_set)
        len_rec = len(recom_set)
        res.append(hit / len_rec)
        n_union += hit
        recommend_sum += len_rec

    return np.mean(res)
    # return n_union / recommend_sum


def get_Recall(rec_list, true_list, true_rel):
    """ 召回率 """
    n_union = 0.
    user_sum = 0.
    recalls = []
    for i in range(len(rec_list)):
        recom_set = set(rec_list[i])
        truth_set = set(true_list[i])
        hit = len(recom_set & truth_set)
        len_truth = len(truth_set)
        recalls.append(hit / len_truth)
        n_union += hit
        user_sum += len_truth

    return np.mean(recalls)
    # return n_union / user_sum

def get_NDCG():
    pass

METRICS = {"recall":get_Recall,
           "precision":get_Precision,
           "ndcg":get_NDCG,
           "mrr":get_MRR,
           "map":get_MAP,
           "ht":get_HR}

def get_ranking_results(df_score, true_list, K=(20,10,5), metrics=["Recall","NDCG"]):

    assert all([metric.lower() in METRICS.keys() for metric in metrics])

    K = sorted(K)[::-1]
    # df_score["y_pred"] = df_score["y_pred"].map(lambda x: np.argsort(x)[::-1])
    df_score["ind"] = df_score.apply(lambda x: np.argsort(x["y_pred"])[::-1], axis=1)
    df_score["item_id_sorted"] = df_score.apply(lambda x: np.array(x["item_id"])[x["ind"]], axis=1)
    df_score["y_pred_sorted"] = df_score.apply(lambda x: np.array(x["y_pred"])[x["ind"]], axis=1)

    df_rec = df_score[["item_id_sorted", "y_pred_sorted"]]

    df_eval = df_rec.join(true_list, on=["user_id"], how="left")
    df_eval = df_eval.loc[~df_eval["y"].isna()]


    true_list = df_eval["item_id"].to_list()
    true_rel = df_eval["y"].to_list()

    results = {}
    for metric in metrics:
        for k in K:
            rec_list = df_eval["item_id_sorted"].map(lambda x: x[:k]).to_list()

            func = METRICS[metric.lower()]
            results[f"{metric}@{k}"] = func(rec_list, true_list, true_rel)







# if __name__ == '__main__':
#     # 推荐列表
#     R = [[3, 10, 15, 12, 17], [20, 15, 18, 14, 30], [2, 5, 7, 8, 15], [56, 14, 25, 12, 19], [21, 24, 36, 54, 45]]
#     # 用户访问列表
#     T = [[12], [3], [5], [14], [20]]
#     # T = [[12, 3, 17, 15], [3], [5, 15, 8], [14], [20, 24]]
#     print(MRR(R, T))
#     print(HitRatio(R, T))
#     print(MAP(R, T))
#     print(Precision(R, T))
#     print(Recall(R, T))