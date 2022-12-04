def get_emb(state_tracker, buffer, indices=None, is_obs=None, obs=None):
    if len(buffer) == 0:
        obs_emb = state_tracker.forward(obs=obs, reset=True)
    else:
        if indices is None:
            indices = buffer.last_index[~buffer[buffer.last_index].done]
            is_obs = False
        if is_obs:
            obs_emb = state_tracker.forward(buffer=buffer, indices=indices, reset=False, is_obs=is_obs)
        else:
            obs_emb = state_tracker.forward(buffer=buffer, indices=indices, reset=False, is_obs=is_obs)
    return obs_emb