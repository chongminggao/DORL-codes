import numpy as np
import torch


def get_emb(state_tracker, buffer, indices=None, is_obs=None, obs=None, remove_recommended_ids=False):
    if len(buffer) == 0:
        obs_emb, recommended_ids = state_tracker.forward(obs=obs, reset=True,
                                                         remove_recommended_ids=remove_recommended_ids)
    else:
        if indices is None:  # collector collects data
            indices = buffer.last_index[~buffer[buffer.last_index].done]
            is_obs = False

        obs_emb, recommended_ids = state_tracker.forward(buffer=buffer, indices=indices, reset=False, is_obs=is_obs,
                                                         remove_recommended_ids=remove_recommended_ids)

    return obs_emb, recommended_ids


def removed_recommended_id_from_embedding(logits, recommended_ids):
    """
    :param logits: Batch * Num_all_items
    :param recommended_ids: Batch * Num_removed
    :return:
    :rtype:
    """

    num_batch, num_action = logits.shape

    indices = np.expand_dims(np.arange(num_action), 0).repeat(num_batch, axis=0)
    indices_torch = torch.from_numpy(indices).to(logits.device)

    if recommended_ids is None:
        return logits, indices_torch

    assert all(recommended_ids[:, -1] == num_action)
    recommended_ids_valid = recommended_ids[:, :-1]
    recommended_ids_valid_torch = torch.LongTensor(recommended_ids_valid).to(device=logits.device)

    mask = torch.ones_like(logits, dtype=torch.bool)
    mask_valid = mask.scatter(1, recommended_ids_valid_torch, 0)

    logits_masked = logits.masked_select(mask_valid).reshape(num_batch, -1)
    indices_masked = indices_torch.masked_select(mask_valid).reshape(num_batch, -1)

    return logits_masked, indices_masked
