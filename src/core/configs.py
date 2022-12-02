def get_features(env, is_userinfo=False):
    if env == "CoatEnv-v0":
        user_features = ["user_id", 'gender_u', 'age', 'location', 'fashioninterest']
        item_features = ['item_id', 'gender_i', "jackettype", 'color', 'onfrontpage']
        reward_features = ["rating"]
    elif env == "KuaiRand-v0":
        user_features = ["user_id", 'user_active_degree', 'is_live_streamer', 'is_video_author',
                         'follow_user_num_range',
                         'fans_user_num_range', 'friend_user_num_range', 'register_days_range'] \
                        + [f'onehot_feat{x}' for x in range(18)]
        if not is_userinfo:
            user_features = ["user_id"]
        item_features = ["item_id"] + ["feat" + str(i) for i in range(3)] + ["duration_normed"]
        reward_features = ["is_click"]
    elif env == "KuaiEnv-v0":
        user_features = ["user_id"]
        item_features = ["item_id"] + ["feat" + str(i) for i in range(4)] + ["duration_normed"]
        reward_features = ["watch_ratio_normed"]
    elif env == "YahooEnv-v0":
        user_features = ["user_id"]
        item_features = ['item_id']
        reward_features = ["rating"]

    return user_features, item_features, reward_features

def get_training_data(env):
    df_train, df_user, df_item, list_feat = None, None, None, None
    if env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        df_train, df_user, df_item, list_feat = CoatEnv.get_df_coat("train.ascii")
    elif env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        df_train, df_user, df_item, list_feat = KuaiRandEnv.get_df_kuairand("train_processed.csv")
    elif env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        df_train, df_user, df_item, list_feat = KuaiEnv.get_df_kuairec("big_matrix_processed.csv")
    elif env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        df_train, df_user, df_item, list_feat = YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-train.txt")

    return df_train, df_user, df_item, list_feat


def get_val_data(env):
    df_train, df_user, df_item, list_feat = None, None, None, None
    if env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        df_val, df_user_val, df_item_val, list_feat = CoatEnv.get_df_coat("test.ascii")
    elif env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        df_val, df_user_val, df_item_val, list_feat = KuaiRandEnv.get_df_kuairand("test_processed.csv")
    elif env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        df_val, df_user_val, df_item_val, list_feat = KuaiEnv.get_df_kuairec("small_matrix_processed.csv")
    elif env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        df_val, df_user_val, df_item_val, list_feat = YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-test.txt")

    return df_val, df_user_val, df_item_val, list_feat

def get_common_args(args):
    env = args.env
    if env == "CoatEnv-v0":
        args.is_userinfo = True
        args.is_binarize = True
        args.need_transform = False
        # args.entropy_on_user = True
        args.entropy_window = [0]
        args.rating_threshold = 4
        args.yfeat = "rating"
    elif env == "KuaiRand-v0":
        args.is_userinfo = False
        args.is_binarize = True
        args.need_transform = False
        # args.entropy_on_user = False
        args.entropy_window = [0, 1, 2]
        args.rating_threshold = 1
        args.yfeat = "is_click"
    elif env == "KuaiEnv-v0":
        args.is_userinfo = False
        args.is_binarize = False
        args.need_transform = True
        # args.entropy_on_user = False
        args.entropy_window = [0,1,2]
        args.yfeat = "watch_ratio_normed"
    elif env == "YahooEnv-v0":
        args.is_userinfo = True
        args.is_binarize = True
        args.need_transform = False
        # args.entropy_on_user = True
        args.entropy_window = [0]
        args.rating_threshold = 4
        args.yfeat = "rating"
    return args

def get_true_env(args, read_user=None):
    if args.env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        mat, df_item, mat_distance = CoatEnv.load_mat()
        kwargs_um = {"mat": mat,
                     "df_item": df_item,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn}
        env = CoatEnv(**kwargs_um)
        env_task_class = CoatEnv
    elif args.env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        mat, mat_distance = YahooEnv.load_mat()
        kwargs_um = {"mat": mat,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn}

        env = YahooEnv(**kwargs_um)
        env_task_class = YahooEnv
    elif args.env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        mat, df_item, mat_distance = KuaiRandEnv.load_mat(args.yfeat, read_user=read_user)
        kwargs_um = {"yname": args.yfeat,
                     "mat": mat,
                     "df_item": df_item,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn}
        env = KuaiRandEnv(**kwargs_um)
        env_task_class = KuaiRandEnv
    elif args.env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        mat, lbe_user, lbe_item, list_feat, df_video_env, df_dist_small = KuaiEnv.load_mat()
        kwargs_um = {"mat": mat,
                     "lbe_user": lbe_user,
                     "lbe_item": lbe_item,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "list_feat": list_feat,
                     "df_video_env": df_video_env,
                     "df_dist_small": df_dist_small}
        env = KuaiEnv(**kwargs_um)
        env_task_class = KuaiEnv
    return env, env_task_class