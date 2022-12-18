# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 5:01 下午
# @Author  : Chongming GAO
# @FileName: evaluation.py
import numpy as np
import torch
from tqdm import tqdm


def get_feat_dominate_dict(env, dataset_val, all_acts, need_transform, item_feat_domination):
    if item_feat_domination is None: # for yahoo
        return dict()
    if need_transform:
        all_acts_origin = env.lbe_item.inverse_transform(all_acts)
    else:
        all_acts_origin = all_acts

    feat_dominate_dict = {}
    recommended_item_features = dataset_val.df_item_val.loc[all_acts_origin]

    if "feat" in item_feat_domination:  # for kuairec and kuairand
        sorted_items = item_feat_domination["feat"]
        dominated_value = sorted_items[0][0]
        recommended_item_features = recommended_item_features.filter(regex="^feat", axis=1)
        feat_flat = recommended_item_features.to_numpy().reshape(-1)
        rate = (feat_flat == dominated_value).sum() / len(recommended_item_features)
        feat_dominate_dict["ifeat_feat"] = rate
    else:  # for coat
        for feat_name, sorted_items in item_feat_domination.items():
            dominated_value = sorted_items[0][0]
            rate = (recommended_item_features[feat_name] == dominated_value).sum() / len(recommended_item_features)
            feat_dominate_dict["ifeat_" + feat_name] = rate

    return feat_dominate_dict
def cannot_overlap_evaluation(model, env, dataset_val, is_softmax, epsilon, is_ucb, k,
                              need_transform, num_trajectory, item_feat_domination, force_length=0):
    cumulative_reward = 0
    total_click_loss = 0
    total_turns = 0

    all_acts = []

    for i in tqdm(range(num_trajectory), desc=f"evaluate static method in {env.__str__()}"):
        user_ori = env.reset()
        if need_transform:
            user = env.lbe_user.inverse_transform(user_ori)[0]
        else:
            user = user_ori

        acts = []
        done = False
        while not done:
            recommended_id_transform, recommended_id_raw, reward_pred = model.recommend_k_item(
                user, dataset_val, k=k, is_softmax=is_softmax, epsilon=epsilon, is_ucb=is_ucb, recommended_ids=acts)
            # if need_transform:
            #     recommendation = env.lbe_item.transform([recommendation])[0]
            acts.append(recommended_id_transform)
            state, reward, done, info = env.step(recommended_id_transform)
            total_turns += 1
            # metric 1
            cumulative_reward += reward
            # metric 2
            click_loss = np.absolute(reward_pred - reward)
            total_click_loss += click_loss

            if done:
                if force_length > 0:  # do not end here
                    env.cur_user = user_ori
                else:
                    break
            if force_length > 0 and len(acts) >= force_length:
                break

        all_acts.extend(acts)



    ctr = cumulative_reward / total_turns
    click_loss = total_click_loss / total_turns

    hit_item = len(set(all_acts))
    num_items = len(dataset_val.df_item_val)
    CV = hit_item / num_items
    CV_turn = hit_item / len(all_acts)

    # eval_result_RL = {"CTR": ctr, "click_loss": click_loss, "trajectory_len": total_turns / num_trajectory,
    #                   "trajectory_reward": cumulative_reward / num_trajectory}
    eval_result_RL = {
        "click_loss": click_loss,
        "CV": f"{CV:.3f}",
        "CV_turn": f"{CV_turn:.3f}",
        "ctr": ctr,
        "len_tra": total_turns / num_trajectory,
        "R_tra": cumulative_reward / num_trajectory}

    feat_dominate_dict = get_feat_dominate_dict(env, dataset_val, all_acts, need_transform, item_feat_domination)
    eval_result_RL.update(feat_dominate_dict)

    eval_result_RL = {f"NX_{force_length}_" + k: v for k, v in eval_result_RL.items()}

    return eval_result_RL




def test_static_model_in_RL_env(model, env, dataset_val, is_softmax=True, epsilon=0, is_ucb=False, k=1,
                                need_transform=False, num_trajectory=200, item_feat_domination=None, force_length=10):
    cumulative_reward = 0
    total_click_loss = 0
    total_turns = 0

    all_acts = []
    for i in tqdm(range(num_trajectory), desc=f"evaluate static method in {env.__str__()}"):
        user = env.reset()
        if need_transform:
            user = env.lbe_user.inverse_transform(user)[0]

        acts = []
        done = False
        while not done:
            recommended_id_transform, recommended_id_raw, reward_pred = model.recommend_k_item(
                user, dataset_val, k=k, is_softmax=is_softmax, epsilon=epsilon, is_ucb=is_ucb, recommended_ids=[])
            # if need_transform:
            #     recommendation = env.lbe_item.transform([recommendation])[0]
            acts.append(recommended_id_transform)
            state, reward, done, info = env.step(recommended_id_transform)
            total_turns += 1
            # metric 1
            cumulative_reward += reward
            # metric 2
            click_loss = np.absolute(reward_pred - reward)
            total_click_loss += click_loss

            if done:
                break
        all_acts.extend(acts)

    ctr = cumulative_reward / total_turns
    click_loss = total_click_loss / total_turns

    hit_item = len(set(all_acts))
    num_items = len(dataset_val.df_item_val)
    CV = hit_item / num_items
    CV_turn = hit_item / len(all_acts)

    # eval_result_RL = {"CTR": ctr, "click_loss": click_loss, "trajectory_len": total_turns / num_trajectory,
    #                   "trajectory_reward": cumulative_reward / num_trajectory}
    eval_result_RL = {"num_test": num_trajectory,
                      "click_loss": click_loss,
                      "CV": f"{CV:.3f}",
                      "CV_turn": f"{CV_turn:.3f}",
                      "ctr": ctr,
                      "len_tra": total_turns / num_trajectory,
                      "R_tra": cumulative_reward / num_trajectory}

    feat_dominate_dict = get_feat_dominate_dict(env, dataset_val, all_acts, need_transform, item_feat_domination)
    eval_result_RL.update(feat_dominate_dict)

    # No overlap and end with the env rule
    eval_result_NX_0 = cannot_overlap_evaluation(model, env, dataset_val, is_softmax, epsilon, is_ucb, k,
                              need_transform, num_trajectory, item_feat_domination, force_length=0)

    # No overlap and end with explicit length
    eval_result_NX_x = cannot_overlap_evaluation(model, env, dataset_val, is_softmax, epsilon, is_ucb, k,
                              need_transform, num_trajectory, item_feat_domination, force_length=force_length)

    eval_result_RL.update(eval_result_NX_0)
    eval_result_RL.update(eval_result_NX_x)

    # if is_ucb:
    #     eval_result_RL.update({"ucb_n": model.n_each})

    return eval_result_RL


def test_taobao(model, env, epsilon=0):
    # test the model in the interactive system
    cumulative_reward = 0
    total_click_loss = 0
    total_turns = 0
    num_trajectory = 100

    for i in range(num_trajectory):
        features = env.reset()
        done = False
        while not done:
            res = model(torch.FloatTensor(features).to(model.device).unsqueeze(0)).to('cpu').squeeze()
            item_feat_predict = res[model.y_index['feat_item'][0]:model.y_index['feat_item'][1]]
            action = item_feat_predict.detach().numpy()

            if epsilon > 0 and np.random.random() < epsilon:
                # Activate epsilon greedy
                action = np.random.random(action.shape)

            reward_pred = res[model.y_index['y'][0]:model.y_index['y'][1]]

            features, reward, done, info = env.step(action)

            total_turns += 1

            # metric 1
            cumulative_reward += reward

            # metric 2
            click_loss = np.absolute(float(reward_pred.detach().numpy()) - reward)
            total_click_loss += click_loss

            if done:
                break

    ctr = cumulative_reward / total_turns  # /10
    click_loss = total_click_loss / total_turns

    # print('CTR: %.2f'.format(ctr))
    eval_result_RL = {"CTR": ctr,
                      "click_loss": click_loss,
                      "trajectory_len": total_turns / num_trajectory,
                      "trajectory_reward": cumulative_reward / num_trajectory}  # /10}

    return eval_result_RL


class Callback_Coverage_Count():
    def __init__(self, test_collector):
        self.test_collector = test_collector
        self.num_items = self.test_collector.env.get_env_attr("mat")[0].shape[1]

    def on_epoch_begin(self, epoch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_end(self, epoch, results=None, **kwargs):
        buffer = self.test_collector.buffer
        live_ind = np.ones([results["n/ep"]], dtype=bool)
        inds = buffer.last_index
        all_acts = []
        while any(live_ind):
            acts = buffer[inds].act
            # print(acts)
            all_acts.extend(acts)

            live_ind = buffer.prev(inds) != inds
            inds = buffer.prev(inds[live_ind])

        hit_item = len(set(all_acts))
        results["CV"] = hit_item / self.num_items
        results["CV_turn"] = hit_item / len(all_acts)
