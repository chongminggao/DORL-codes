

#python run_A2CPolicy_coat.py --cuda 2  --read_message "pointneg" --window 1 --message "win1" &
#python run_A2CPolicy_coat.py --cuda 2  --read_message "pointneg" --window 2 --message "win2" &
#python run_A2CPolicy_coat.py --cuda 7  --read_message "pointneg" --window 3 --message "win3" &
#python run_A2CPolicy_coat.py --cuda 7  --read_message "pointneg" --window 4 --message "win4" &




#python run_A2CPolicy_coat2.py --cuda 1  --read_message "pointneg" --window 1 --message "v2win1" &
#python run_A2CPolicy_coat2.py --cuda 1  --read_message "pointneg" --window 2 --message "v2win2" &
#python run_A2CPolicy_coat2.py --cuda 1  --read_message "pointneg" --window 3 --message "v2win3" &
#python run_A2CPolicy_coat2.py --cuda 1  --read_message "pointneg" --window 4 --message "v2win4" &
#
#python run_A2CPolicy_coat2.py --cuda 3  --read_message "pointneg" --window 1 --message "v2win1_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 3  --read_message "pointneg" --window 2 --message "v2win2_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 3  --read_message "pointneg" --window 3 --message "v2win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 3  --read_message "pointneg" --window 4 --message "v2win4_user" --is_use_userEmbedding &


#python run_A2CPolicy_coat2.py --cuda 0  --read_message "pointneg" --window 5 --message "v2win5" &
#python run_A2CPolicy_coat2.py --cuda 2  --read_message "pointneg" --window 6 --message "v2win6" &
#python run_A2CPolicy_coat2.py --cuda 4  --read_message "pointneg" --window 7 --message "v2win7" &
#python run_A2CPolicy_coat2.py --cuda 6  --read_message "pointneg" --window 8 --message "v2win8" &
#
#python run_A2CPolicy_coat2.py --cuda 0  --read_message "pointneg" --window 5 --message "v2win5_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 2  --read_message "pointneg" --window 6 --message "v2win6_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 4  --read_message "pointneg" --window 7 --message "v2win7_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 6  --read_message "pointneg" --window 8 --message "v2win8_user" --is_use_userEmbedding &

#python run_A2CPolicy_coat2.py --cuda 7  --read_message "pointneg" --window 1 --message "freeze_v2win1" --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 7  --read_message "pointneg" --window 2 --message "freeze_v2win2" --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 7  --read_message "pointneg" --window 1 --message "freeze_v2win1_user" --is_use_userEmbedding --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 7  --read_message "pointneg" --window 2 --message "freeze_v2win2_user" --is_use_userEmbedding --is_freeze_emb &


#python run_A2CPolicy_coat2.py --cuda 0  --lambda_variance 0.1 --read_message "pointneg" --window 2 --message "var0.1_win2" &
#python run_A2CPolicy_coat2.py --cuda 1  --lambda_variance 0.5 --read_message "pointneg" --window 2 --message "var0.5_win2" &
#python run_A2CPolicy_coat2.py --cuda 2  --lambda_variance 1 --read_message "pointneg" --window 2 --message "var1_win2" &
#python run_A2CPolicy_coat2.py --cuda 3  --lambda_variance 5 --read_message "pointneg" --window 2 --message "var5_win2" &
#python run_A2CPolicy_coat2.py --cuda 4  --lambda_variance 10 --read_message "pointneg" --window 2 --message "var10_win2" &
#
#python run_A2CPolicy_coat2.py --cuda 5  --lambda_variance 0.1 --read_message "pointneg" --window 3 --message "var0.1_win3" &
#python run_A2CPolicy_coat2.py --cuda 6  --lambda_variance 0.5 --read_message "pointneg" --window 3 --message "var0.5_win3" &
#python run_A2CPolicy_coat2.py --cuda 7  --lambda_variance 1 --read_message "pointneg" --window 3 --message "var1_win3" &
#python run_A2CPolicy_coat2.py --cuda 0  --lambda_variance 5 --read_message "pointneg" --window 3 --message "var5_win3" &
#python run_A2CPolicy_coat2.py --cuda 1  --lambda_variance 10 --read_message "pointneg" --window 3 --message "var10_win3" &
#
#python run_A2CPolicy_coat2.py --cuda 2  --lambda_variance 0.1 --read_message "pointneg" --window 2 --message "var0.1_win2_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 3  --lambda_variance 0.5 --read_message "pointneg" --window 2 --message "var0.5_win2_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 4  --lambda_variance 1 --read_message "pointneg" --window 2 --message "var1_win2_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 5  --lambda_variance 5 --read_message "pointneg" --window 2 --message "var5_win2_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 6  --lambda_variance 10 --read_message "pointneg" --window 2 --message "var10_win2_user" --is_use_userEmbedding &
#
#python run_A2CPolicy_coat2.py --cuda 7  --lambda_variance 0.1 --read_message "pointneg" --window 3 --message "var0.1_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 0  --lambda_variance 0.5 --read_message "pointneg" --window 3 --message "var0.5_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 1  --lambda_variance 1 --read_message "pointneg" --window 3 --message "var1_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 2  --lambda_variance 5 --read_message "pointneg" --window 3 --message "var5_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 3  --lambda_variance 10 --read_message "pointneg" --window 3 --message "var10_win3_user" --is_use_userEmbedding &


#python run_A2CPolicy_coat2.py --cuda 0  --lambda_variance 0.1 --read_message "pointneg" --window 2 --message "var0.1_win2_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 1  --lambda_variance 0.2 --read_message "pointneg" --window 2 --message "var0.2_win2_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 2  --lambda_variance 0.3 --read_message "pointneg" --window 2 --message "var0.3_win2_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 3  --lambda_variance 0.4 --read_message "pointneg" --window 2 --message "var0.4_win2_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 4  --lambda_variance 0.5 --read_message "pointneg" --window 2 --message "var0.5_win2_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 5  --lambda_variance 0.6 --read_message "pointneg" --window 2 --message "var0.6_win2_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 6  --lambda_variance 0.7 --read_message "pointneg" --window 2 --message "var0.7_win2_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 7  --lambda_variance 0.8 --read_message "pointneg" --window 2 --message "var0.8_win2_user" --is_use_userEmbedding &


#python run_A2CPolicy_coat2.py --cuda 0  --lambda_variance 0.1 --read_message "pointneg" --window 3 --message "var0.1_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 1  --lambda_variance 0.2 --read_message "pointneg" --window 3 --message "var0.2_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 2  --lambda_variance 0.3 --read_message "pointneg" --window 3 --message "var0.3_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 3  --lambda_variance 0.4 --read_message "pointneg" --window 3 --message "var0.4_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 4  --lambda_variance 0.5 --read_message "pointneg" --window 3 --message "var0.5_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 5  --lambda_variance 0.6 --read_message "pointneg" --window 3 --message "var0.6_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 6  --lambda_variance 0.7 --read_message "pointneg" --window 3 --message "var0.7_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 7  --lambda_variance 0.8 --read_message "pointneg" --window 3 --message "var0.8_win3_user" --is_use_userEmbedding &
#
#
#python run_A2CPolicy_coat2.py --cuda 0  --lambda_variance 0.1 --read_message "pointneg" --window 3 --message "var0.1_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 1  --lambda_variance 0.2 --read_message "pointneg" --window 3 --message "var0.2_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 2  --lambda_variance 0.3 --read_message "pointneg" --window 3 --message "var0.3_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 3  --lambda_variance 0.4 --read_message "pointneg" --window 3 --message "var0.4_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 4  --lambda_variance 0.5 --read_message "pointneg" --window 3 --message "var0.5_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 5  --lambda_variance 0.6 --read_message "pointneg" --window 3 --message "var0.6_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 6  --lambda_variance 0.7 --read_message "pointneg" --window 3 --message "var0.7_win3_user" --is_use_userEmbedding &
#python run_A2CPolicy_coat2.py --cuda 7  --lambda_variance 0.8 --read_message "pointneg" --window 3 --message "var0.8_win3_user" --is_use_userEmbedding &


#python run_A2CPolicy_coat2.py --cuda 1  --read_message "pointneg" --window 1 --message "freeze_v2win1" --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 2  --read_message "pointneg" --window 2 --message "freeze_v2win2" --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 3  --read_message "pointneg" --window 3 --message "freeze_v2win3" --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 4  --read_message "pointneg" --window 1 --message "freeze_v2win1_user" --is_use_userEmbedding --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 5  --read_message "pointneg" --window 2 --message "freeze_v2win2_user" --is_use_userEmbedding --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 7  --read_message "pointneg" --window 3 --message "freeze_v2win3_user" --is_use_userEmbedding --is_freeze_emb &


#python run_A2CPolicy_coat2.py --cuda 1  --read_message "pointneg" --window 1 --message "nofreeze_v2win1" --no_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 2  --read_message "pointneg" --window 2 --message "nofreeze_v2win2" --no_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 3  --read_message "pointneg" --window 3 --message "nofreeze_v2win3" --no_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 4  --read_message "pointneg" --window 1 --message "nofreeze_v2win1_user" --is_use_userEmbedding --no_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 5  --read_message "pointneg" --window 2 --message "nofreeze_v2win2_user" --is_use_userEmbedding --no_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 7  --read_message "pointneg" --window 3 --message "nofreeze_v2win3_user" --is_use_userEmbedding --no_freeze_emb &


#python run_A2CPolicy_coat2.py --cuda 1  --read_message "pointneg" --window 4 --message "freeze_v2win4" --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 2  --read_message "pointneg" --window 4 --message "freeze_v2win4_user" --is_use_userEmbedding --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 3  --read_message "pointneg" --window 4 --message "nofreeze_v2win4" --no_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 4  --read_message "pointneg" --window 4 --message "nofreeze_v2win4_user" --is_use_userEmbedding --no_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 1  --read_message "pointneg" --window 5 --message "freeze_v2win5" --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 2  --read_message "pointneg" --window 5 --message "freeze_v2win5_user" --is_use_userEmbedding --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 3  --read_message "pointneg" --window 5 --message "nofreeze_v2win5" --no_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 4  --read_message "pointneg" --window 5 --message "nofreeze_v2win5_user" --is_use_userEmbedding --no_freeze_emb &




#python run_A2CPolicy_coat2.py --cuda 1  --lambda_variance 0.1 --read_message "pointneg" --window 3 --message "var0.1_win3_nofreeze_100k" --no_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 2  --lambda_variance 0.5 --read_message "pointneg" --window 3 --message "var0.5_win3_nofreeze_100k" --no_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 3  --lambda_variance 1 --read_message "pointneg" --window 3 --message "var1_win3_nofreeze_100k" --no_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 4  --lambda_variance 5 --read_message "pointneg" --window 3 --message "var5_win3_nofreeze_100k" --no_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 5  --lambda_variance 10 --read_message "pointneg" --window 3 --message "var10_win3_nofreeze_100k" --no_freeze_emb &
#
#
#python run_A2CPolicy_coat2.py --cuda 1  --lambda_variance 0.1 --read_message "pointneg" --window 3 --message "var0.1_win3_isfreeze_100k" --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 2  --lambda_variance 0.5 --read_message "pointneg" --window 3 --message "var0.5_win3_isfreeze_100k" --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 3  --lambda_variance 1 --read_message "pointneg" --window 3 --message "var1_win3_isfreeze_100k" --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 4  --lambda_variance 5 --read_message "pointneg" --window 3 --message "var5_win3_isfreeze_100k" --is_freeze_emb &
#python run_A2CPolicy_coat2.py --cuda 5  --lambda_variance 10 --read_message "pointneg" --window 3 --message "var10_win3_isfreeze_100k" --is_freeze_emb &





#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 6  --lambda_variance 1 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var1_ent1_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 6  --lambda_variance 1 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var1_ent1_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 6  --lambda_variance 1 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var1_ent1_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 6  --lambda_variance 1 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var1_ent1_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 7  --lambda_variance 1 --lambda_entropy 5 --window_size 3 --read_message "pointneg"  --message "var1_ent5_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 7  --lambda_variance 1 --lambda_entropy 5 --window_size 3 --read_message "pointneg"  --message "var1_ent5_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 7  --lambda_variance 1 --lambda_entropy 5 --window_size 3 --read_message "pointneg"  --message "var1_ent5_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 7  --lambda_variance 1 --lambda_entropy 5 --window_size 3 --read_message "pointneg"  --message "var1_ent5_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2  --lambda_variance 1 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var1_ent10_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 2  --lambda_variance 1 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var1_ent10_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 2  --lambda_variance 1 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var1_ent10_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 2  --lambda_variance 1 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var1_ent10_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3  --lambda_variance 1 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var1_ent1_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 3  --lambda_variance 1 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var1_ent1_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 3  --lambda_variance 1 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var1_ent1_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 3  --lambda_variance 1 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var1_ent1_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4  --lambda_variance 1 --lambda_entropy 5 --window_size 3 --read_message "pointneg"  --message "var5_ent1_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 4  --lambda_variance 1 --lambda_entropy 5 --window_size 3 --read_message "pointneg"  --message "var5_ent1_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 4  --lambda_variance 1 --lambda_entropy 5 --window_size 3 --read_message "pointneg"  --message "var5_ent1_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 4  --lambda_variance 1 --lambda_entropy 5 --window_size 3 --read_message "pointneg"  --message "var5_ent1_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 5  --lambda_variance 1 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var10_ent1_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 5  --lambda_variance 1 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var10_ent1_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 5  --lambda_variance 1 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var10_ent1_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 5  --lambda_variance 1 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var10_ent1_win3"  &



#python run_Policy_Main.py --env CoatEnv-v0  --cuda 6  --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "newR"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 6  --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "newR"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 2  --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "newR"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 3  --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "newR"  &
#
#python run_Policy_BCQ.py --env CoatEnv-v0  --cuda 6 --window_size 3  --read_message "pointneg"  --message "debug" &
#python run_Policy_BCQ.py --env YahooEnv-v0 --cuda 6 --window_size 3  --read_message "pointneg"  --message "debug" &
#python run_Policy_BCQ.py --env KuaiRand-v0 --cuda 6 --window_size 3  --read_message "pointneg"  --message "debug" &
#python run_Policy_BCQ.py --env KuaiEnv-v0  --cuda 6 --window_size 3  --read_message "pointneg"  --message "debug" &
#
#python run_Policy_CQL.py --env CoatEnv-v0  --cuda 5 --window_size 3  --read_message "pointneg"  --message "CQL" &
#python run_Policy_CQL.py --env YahooEnv-v0 --cuda 5 --window_size 3  --read_message "pointneg"  --message "CQL" &
#python run_Policy_CQL.py --env KuaiRand-v0 --cuda 5 --window_size 3  --read_message "pointneg"  --message "CQL" &
#python run_Policy_CQL.py --env KuaiEnv-v0  --cuda 5 --window_size 3  --read_message "pointneg"  --message "CQL" &
#
#python run_Policy_CRR.py --env CoatEnv-v0  --cuda 5 --window_size 3  --read_message "pointneg"  --message "debug" &
#python run_Policy_CRR.py --env YahooEnv-v0 --cuda 5 --window_size 3  --read_message "pointneg"  --message "debug" &
#python run_Policy_CRR.py --env KuaiRand-v0 --cuda 5 --window_size 3  --read_message "pointneg"  --message "debug" &
#python run_Policy_CRR.py --env KuaiEnv-v0  --cuda 5 --window_size 3  --read_message "pointneg"  --message "debug" &



#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "noR"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 0  --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "noR"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 0  --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "noR"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 0  --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "noR"  &

#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0_ent0_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 0  --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0_ent0_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 0  --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0_ent0_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 0  --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0_ent0_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --lambda_variance 0 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var0_ent1_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 0  --lambda_variance 0 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var0_ent1_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 0  --lambda_variance 0 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var0_ent1_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 0  --lambda_variance 0 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var0_ent1_win3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --lambda_variance 1 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var1_ent0_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 1  --lambda_variance 1 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var1_ent0_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 1  --lambda_variance 1 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var1_ent0_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 1  --lambda_variance 1 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var1_ent0_win3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --lambda_variance 1 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var1_ent1_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 1  --lambda_variance 1 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var1_ent1_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 1  --lambda_variance 1 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var1_ent1_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 1  --lambda_variance 1 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var1_ent1_win3"  &
#
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2  --lambda_variance 0 --lambda_entropy 5 --window_size 3 --read_message "pointneg"  --message "var0_ent5_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 2  --lambda_variance 0 --lambda_entropy 5 --window_size 3 --read_message "pointneg"  --message "var0_ent5_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 2  --lambda_variance 0 --lambda_entropy 5 --window_size 3 --read_message "pointneg"  --message "var0_ent5_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 2  --lambda_variance 0 --lambda_entropy 5 --window_size 3 --read_message "pointneg"  --message "var0_ent5_win3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2  --lambda_variance 5 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var5_ent0_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 2  --lambda_variance 5 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var5_ent0_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 2  --lambda_variance 5 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var5_ent0_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 2  --lambda_variance 5 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var5_ent0_win3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4  --lambda_variance 5 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var5_ent1_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 4  --lambda_variance 5 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var5_ent1_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 4  --lambda_variance 5 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var5_ent1_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 4  --lambda_variance 5 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var5_ent1_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4  --lambda_variance 0 --lambda_entropy 0.1 --window_size 3 --read_message "pointneg"  --message "var0_ent0.1_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 4  --lambda_variance 0 --lambda_entropy 0.1 --window_size 3 --read_message "pointneg"  --message "var0_ent0.1_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 4  --lambda_variance 0 --lambda_entropy 0.1 --window_size 3 --read_message "pointneg"  --message "var0_ent0.1_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 4  --lambda_variance 0 --lambda_entropy 0.1 --window_size 3 --read_message "pointneg"  --message "var0_ent0.1_win3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 5  --lambda_variance 0.1 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.1_ent0_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 5  --lambda_variance 0.1 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.1_ent0_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 5  --lambda_variance 0.1 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.1_ent0_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 5  --lambda_variance 0.1 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.1_ent0_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 5  --lambda_variance 0 --lambda_entropy 0.5 --window_size 3 --read_message "pointneg"  --message "var0_ent0.5_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 5  --lambda_variance 0 --lambda_entropy 0.5 --window_size 3 --read_message "pointneg"  --message "var0_ent0.5_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 5  --lambda_variance 0 --lambda_entropy 0.5 --window_size 3 --read_message "pointneg"  --message "var0_ent0.5_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 5  --lambda_variance 0 --lambda_entropy 0.5 --window_size 3 --read_message "pointneg"  --message "var0_ent0.5_win3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 6  --lambda_variance 0.5 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.5_ent0_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 6  --lambda_variance 0.5 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.5_ent0_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 6  --lambda_variance 0.5 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.5_ent0_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 6  --lambda_variance 0.5 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.5_ent0_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 6  --lambda_variance 0 --lambda_entropy 0.01 --window_size 3 --read_message "pointneg"  --message "var0_ent0.01_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 6  --lambda_variance 0 --lambda_entropy 0.01 --window_size 3 --read_message "pointneg"  --message "var0_ent0.01_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 6  --lambda_variance 0 --lambda_entropy 0.01 --window_size 3 --read_message "pointneg"  --message "var0_ent0.01_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 6  --lambda_variance 0 --lambda_entropy 0.01 --window_size 3 --read_message "pointneg"  --message "var0_ent0.01_win3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 7  --lambda_variance 0.01 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.01_ent0_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 7  --lambda_variance 0.01 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.01_ent0_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 7  --lambda_variance 0.01 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.01_ent0_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 7  --lambda_variance 0.01 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.01_ent0_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 7  --lambda_variance 0 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var0_ent10_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 7  --lambda_variance 0 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var0_ent10_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 7  --lambda_variance 0 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var0_ent10_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 7  --lambda_variance 0 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var0_ent10_win3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3  --lambda_variance 10 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var10_ent0_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 3  --lambda_variance 10 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var10_ent0_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 3  --lambda_variance 10 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var10_ent0_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 3  --lambda_variance 10 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var10_ent0_win3"  &

#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4  --lambda_variance 0 --lambda_entropy 0.1 --window_size 3 --read_message "pointneg"  --message "var0_ent0.1_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 4  --lambda_variance 0 --lambda_entropy 0.1 --window_size 3 --read_message "pointneg"  --message "var0_ent0.1_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 4  --lambda_variance 0 --lambda_entropy 0.1 --window_size 3 --read_message "pointneg"  --message "var0_ent0.1_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 4  --lambda_variance 0 --lambda_entropy 0.1 --window_size 3 --read_message "pointneg"  --message "var0_ent0.1_win3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 5  --lambda_variance 0.1 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.1_ent0_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 5  --lambda_variance 0.1 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.1_ent0_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 5  --lambda_variance 0.1 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.1_ent0_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 5  --lambda_variance 0.1 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.1_ent0_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 5  --lambda_variance 0 --lambda_entropy 0.5 --window_size 3 --read_message "pointneg"  --message "var0_ent0.5_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 5  --lambda_variance 0 --lambda_entropy 0.5 --window_size 3 --read_message "pointneg"  --message "var0_ent0.5_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 5  --lambda_variance 0 --lambda_entropy 0.5 --window_size 3 --read_message "pointneg"  --message "var0_ent0.5_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 5  --lambda_variance 0 --lambda_entropy 0.5 --window_size 3 --read_message "pointneg"  --message "var0_ent0.5_win3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 6  --lambda_variance 0.5 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.5_ent0_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 6  --lambda_variance 0.5 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.5_ent0_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 6  --lambda_variance 0.5 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.5_ent0_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 6  --lambda_variance 0.5 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.5_ent0_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 6  --lambda_variance 0 --lambda_entropy 0.01 --window_size 3 --read_message "pointneg"  --message "var0_ent0.01_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 6  --lambda_variance 0 --lambda_entropy 0.01 --window_size 3 --read_message "pointneg"  --message "var0_ent0.01_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 6  --lambda_variance 0 --lambda_entropy 0.01 --window_size 3 --read_message "pointneg"  --message "var0_ent0.01_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 6  --lambda_variance 0 --lambda_entropy 0.01 --window_size 3 --read_message "pointneg"  --message "var0_ent0.01_win3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 7  --lambda_variance 0.01 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.01_ent0_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 7  --lambda_variance 0.01 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.01_ent0_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 7  --lambda_variance 0.01 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.01_ent0_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 7  --lambda_variance 0.01 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var0.01_ent0_win3"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 7  --lambda_variance 0 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var0_ent10_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 7  --lambda_variance 0 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var0_ent10_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 7  --lambda_variance 0 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var0_ent10_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 7  --lambda_variance 0 --lambda_entropy 10 --window_size 3 --read_message "pointneg"  --message "var0_ent10_win3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3  --lambda_variance 10 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var10_ent0_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 3  --lambda_variance 10 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var10_ent0_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 3  --lambda_variance 10 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var10_ent0_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 3  --lambda_variance 10 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var10_ent0_win3"  &

#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3  --lambda_variance 10 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var10_ent0_win3"  &
#python run_Policy_Main.py --env YahooEnv-v0 --cuda 3  --lambda_variance 10 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var10_ent0_win3"  &
#python run_Policy_Main.py --env KuaiRand-v0 --cuda 3  --lambda_variance 10 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var10_ent0_win3"  &
#python run_Policy_Main.py --env KuaiEnv-v0  --cuda 3  --lambda_variance 10 --lambda_entropy 1 --window_size 3 --read_message "pointneg"  --message "var10_ent0_win3"  &





#python run_Policy_SQN.py --env CoatEnv-v0  --cuda 4 --window_sqn 10 --which_tracker caser  --message "sqn_caser_win10" &
#python run_Policy_SQN.py --env YahooEnv-v0 --cuda 4 --window_sqn 10 --which_tracker caser  --message "sqn_caser_win10" &
#python run_Policy_SQN.py --env KuaiRand-v0 --cuda 4 --window_sqn 10 --which_tracker caser  --message "sqn_caser_win10" &
#python run_Policy_SQN.py --env KuaiEnv-v0  --cuda 4 --window_sqn 10 --which_tracker caser  --message "sqn_caser_win10" &
#
#python run_Policy_SQN.py --env CoatEnv-v0  --cuda 5 --window_sqn 5 --which_tracker caser  --message "sqn_caser_win5" &
#python run_Policy_SQN.py --env YahooEnv-v0 --cuda 5 --window_sqn 5 --which_tracker caser  --message "sqn_caser_win5" &
#python run_Policy_SQN.py --env KuaiRand-v0 --cuda 5 --window_sqn 5 --which_tracker caser  --message "sqn_caser_win5" &
#python run_Policy_SQN.py --env KuaiEnv-v0  --cuda 5 --window_sqn 5 --which_tracker caser  --message "sqn_caser_win5" &
#
#python run_Policy_SQN.py --env CoatEnv-v0  --cuda 6 --window_sqn 3 --which_tracker gru  --message "sqn_gru_win3" &
#python run_Policy_SQN.py --env YahooEnv-v0 --cuda 6 --window_sqn 3 --which_tracker gru  --message "sqn_gru_win3" &
#python run_Policy_SQN.py --env KuaiRand-v0 --cuda 6 --window_sqn 3 --which_tracker gru  --message "sqn_gru_win3" &
#python run_Policy_SQN.py --env KuaiEnv-v0  --cuda 6 --window_sqn 3 --which_tracker gru  --message "sqn_gru_win3" &
#
#python run_Policy_SQN.py --env CoatEnv-v0  --cuda 7 --window_sqn 5 --which_tracker gru  --message "sqn_gru_win5" &
#python run_Policy_SQN.py --env YahooEnv-v0 --cuda 7 --window_sqn 5 --which_tracker gru  --message "sqn_gru_win5" &
#python run_Policy_SQN.py --env KuaiRand-v0 --cuda 7 --window_sqn 5 --which_tracker gru  --message "sqn_gru_win5" &
#python run_Policy_SQN.py --env KuaiEnv-v0  --cuda 7 --window_sqn 5 --which_tracker gru  --message "sqn_gru_win5" &
#
#python run_Policy_SQN.py --env CoatEnv-v0  --cuda 4 --window_sqn 10 --which_tracker gru  --message "sqn_gru_win10" &
#python run_Policy_SQN.py --env YahooEnv-v0 --cuda 4 --window_sqn 10 --which_tracker gru  --message "sqn_gru_win10" &
#python run_Policy_SQN.py --env KuaiRand-v0 --cuda 4 --window_sqn 10 --which_tracker gru  --message "sqn_gru_win10" &
#python run_Policy_SQN.py --env KuaiEnv-v0  --cuda 4 --window_sqn 10 --which_tracker gru  --message "sqn_gru_win10" &
#
#python run_Policy_SQN.py --env CoatEnv-v0  --cuda 5 --window_sqn 3 --which_tracker sasrec  --message "sqn_sasrec_win3" &
#python run_Policy_SQN.py --env YahooEnv-v0 --cuda 5 --window_sqn 3 --which_tracker sasrec  --message "sqn_sasrec_win3" &
#python run_Policy_SQN.py --env KuaiRand-v0 --cuda 5 --window_sqn 3 --which_tracker sasrec  --message "sqn_sasrec_win3" &
#python run_Policy_SQN.py --env KuaiEnv-v0  --cuda 5 --window_sqn 3 --which_tracker sasrec  --message "sqn_sasrec_win3" &
#
#python run_Policy_SQN.py --env CoatEnv-v0  --cuda 6 --window_sqn 5 --which_tracker sasrec  --message "sqn_sasrec_win5" &
#python run_Policy_SQN.py --env YahooEnv-v0 --cuda 6 --window_sqn 5 --which_tracker sasrec  --message "sqn_sasrec_win5" &
#python run_Policy_SQN.py --env KuaiRand-v0 --cuda 6 --window_sqn 5 --which_tracker sasrec  --message "sqn_sasrec_win5" &
#python run_Policy_SQN.py --env KuaiEnv-v0  --cuda 6 --window_sqn 5 --which_tracker sasrec  --message "sqn_sasrec_win5" &
#
#python run_Policy_SQN.py --env CoatEnv-v0  --cuda 7 --window_sqn 10 --which_tracker sasrec  --message "sqn_sasrec_win10" &
#python run_Policy_SQN.py --env YahooEnv-v0 --cuda 7 --window_sqn 10 --which_tracker sasrec  --message "sqn_sasrec_win10" &
#python run_Policy_SQN.py --env KuaiRand-v0 --cuda 7 --window_sqn 10 --which_tracker sasrec  --message "sqn_sasrec_win10" &
#python run_Policy_SQN.py --env KuaiEnv-v0  --cuda 7 --window_sqn 10 --which_tracker sasrec  --message "sqn_sasrec_win10" &







#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --reward_handle "no" --lambda_variance 0     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var0"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --reward_handle "no" --lambda_variance 0.001 --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var1"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --reward_handle "no" --lambda_variance 0.005 --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var2"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --reward_handle "no" --lambda_variance 0.01  --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --reward_handle "no" --lambda_variance 0.05  --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var4"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --reward_handle "no" --lambda_variance 0.1   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var5"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --reward_handle "no" --lambda_variance 0.5   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var6"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --reward_handle "no" --lambda_variance 1     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var7"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --reward_handle "no" --lambda_variance 5     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var8"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --reward_handle "no" --lambda_variance 10    --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var9"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --reward_handle "no" --lambda_variance 50    --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var10"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --reward_handle "no" --lambda_variance 100   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var11"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2 --reward_handle "cat"  --lambda_variance 0     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var0"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2 --reward_handle "cat"  --lambda_variance 0.001 --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var1"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2 --reward_handle "cat"  --lambda_variance 0.005 --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var2"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2 --reward_handle "cat"  --lambda_variance 0.01  --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2 --reward_handle "cat"  --lambda_variance 0.05  --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var4"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2 --reward_handle "cat"  --lambda_variance 0.1   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var5"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3 --reward_handle "cat"  --lambda_variance 0.5   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var6"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3 --reward_handle "cat"  --lambda_variance 1     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var7"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3 --reward_handle "cat"  --lambda_variance 5     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var8"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3 --reward_handle "cat"  --lambda_variance 10    --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var9"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3 --reward_handle "cat"  --lambda_variance 50    --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var10"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3 --reward_handle "cat"  --lambda_variance 100   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var11"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var0"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0.001 --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var1"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0.005 --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var2"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0.01  --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0.05  --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var4"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0.1   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var5"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0.5   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var6"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 1     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var7"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 5     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var8"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 10    --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var9"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 50    --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var10"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 100   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var11"  &
#
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var0"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0.001 --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var1"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0.005 --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var2"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0.01  --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var3"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0.05  --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var4"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0.1   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var5"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0.5   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var6"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 1     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var7"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 5     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var8"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 10    --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var9"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 50    --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var10"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 100   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_no_var11"  &
#
#
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var0"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0.001 --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var1"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0.005 --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var2"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0.01  --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var3"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0.05  --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var4"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0.1   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var5"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0.5   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var6"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 1     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var7"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 5     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var8"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 10    --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var9"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 50    --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var10"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 100   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat_var11"  &
#
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var0"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0.001 --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var1"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0.005 --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var2"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0.01  --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var3"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0.05  --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var4"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0.1   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var5"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0.5   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var6"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 1     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var7"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 5     --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var8"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 10    --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var9"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 50    --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var10"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 100   --lambda_entropy 0 --window_size 2 --read_message "pointneg"  --message "win2_cat2_var11"  &




#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --reward_handle "no" --lambda_variance 0     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var0"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --reward_handle "no" --lambda_variance 0.001 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var1"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --reward_handle "no" --lambda_variance 0.005 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var2"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --reward_handle "no" --lambda_variance 0.01  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --reward_handle "no" --lambda_variance 0.05  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var4"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --reward_handle "no" --lambda_variance 0.1   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var5"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --reward_handle "no" --lambda_variance 0.5   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var6"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --reward_handle "no" --lambda_variance 1     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var7"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --reward_handle "no" --lambda_variance 5     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var8"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --reward_handle "no" --lambda_variance 10    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var9"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --reward_handle "no" --lambda_variance 50    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var10"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --reward_handle "no" --lambda_variance 100   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var11"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2 --reward_handle "cat"  --lambda_variance 0     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var0"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2 --reward_handle "cat"  --lambda_variance 0.001 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var1"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2 --reward_handle "cat"  --lambda_variance 0.005 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var2"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2 --reward_handle "cat"  --lambda_variance 0.01  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2 --reward_handle "cat"  --lambda_variance 0.05  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var4"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2 --reward_handle "cat"  --lambda_variance 0.1   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var5"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3 --reward_handle "cat"  --lambda_variance 0.5   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var6"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3 --reward_handle "cat"  --lambda_variance 1     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var7"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3 --reward_handle "cat"  --lambda_variance 5     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var8"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3 --reward_handle "cat"  --lambda_variance 10    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var9"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3 --reward_handle "cat"  --lambda_variance 50    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var10"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 3 --reward_handle "cat"  --lambda_variance 100   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var11"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var0"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0.001 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var1"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0.005 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var2"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0.01  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0.05  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var4"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0.1   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var5"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 0.5   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var6"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 1     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var7"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 5     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var8"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 10    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var9"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 50    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var10"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4 --reward_handle "cat2"  --lambda_variance 100   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var11"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var0"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0.001 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var1"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0.005 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var2"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0.01  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var3"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0.05  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var4"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0.1   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var5"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 0.5   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var6"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 1     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var7"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 5     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var8"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 10    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var9"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 50    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var10"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 5  --reward_handle "no" --lambda_variance 100   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_no_var11"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var0"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0.001 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var1"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0.005 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var2"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0.01  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var3"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0.05  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var4"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0.1   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var5"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 0.5   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var6"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 1     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var7"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 5     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var8"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 10    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var9"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 50    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var10"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 7 --reward_handle "cat"  --lambda_variance 100   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat_var11"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var0"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0.001 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var1"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0.005 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var2"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0.01  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var3"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0.05  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var4"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0.1   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var5"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 0.5   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var6"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 1     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var7"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 5     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var8"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 10    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var9"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 50    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var10"  &
#python run_Policy_Main.py --env YahooEnv-v0  --cuda 6 --reward_handle "cat2"  --lambda_variance 100   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "win5_cat2_var11"  &


#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --which_tracker caser --reward_handle "no" --lambda_variance 0     --lambda_entropy 0 --window_size 10 --read_message "pointneg"  --message "caser_win10_no_var0"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --which_tracker caser --reward_handle "no" --lambda_variance 0.001 --lambda_entropy 0 --window_size 10 --read_message "pointneg"  --message "caser_win10_no_var1"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --which_tracker caser --reward_handle "no" --lambda_variance 0.005 --lambda_entropy 0 --window_size 10 --read_message "pointneg"  --message "caser_win10_no_var2"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --which_tracker caser --reward_handle "no" --lambda_variance 0.01  --lambda_entropy 0 --window_size 10 --read_message "pointneg"  --message "caser_win10_no_var3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --which_tracker caser --reward_handle "no" --lambda_variance 0.05  --lambda_entropy 0 --window_size 10 --read_message "pointneg"  --message "caser_win10_no_var4"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 0  --which_tracker caser --reward_handle "no" --lambda_variance 0.1   --lambda_entropy 0 --window_size 10 --read_message "pointneg"  --message "caser_win10_no_var5"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --which_tracker caser --reward_handle "no" --lambda_variance 0.5   --lambda_entropy 0 --window_size 10 --read_message "pointneg"  --message "caser_win10_no_var6"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --which_tracker caser --reward_handle "no" --lambda_variance 1     --lambda_entropy 0 --window_size 10 --read_message "pointneg"  --message "caser_win10_no_var7"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --which_tracker caser --reward_handle "no" --lambda_variance 5     --lambda_entropy 0 --window_size 10 --read_message "pointneg"  --message "caser_win10_no_var8"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --which_tracker caser --reward_handle "no" --lambda_variance 10    --lambda_entropy 0 --window_size 10 --read_message "pointneg"  --message "caser_win10_no_var9"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --which_tracker caser --reward_handle "no" --lambda_variance 50    --lambda_entropy 0 --window_size 10 --read_message "pointneg"  --message "caser_win10_no_var10"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 1  --which_tracker caser --reward_handle "no" --lambda_variance 100   --lambda_entropy 0 --window_size 10 --read_message "pointneg"  --message "caser_win10_no_var11"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2  --which_tracker caser --reward_handle "no" --lambda_variance 0     --lambda_entropy 0 --window_size 8 --read_message "pointneg"  --message "caser_win8_no_var0"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2  --which_tracker caser --reward_handle "no" --lambda_variance 0.001 --lambda_entropy 0 --window_size 8 --read_message "pointneg"  --message "caser_win8_no_var1"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2  --which_tracker caser --reward_handle "no" --lambda_variance 0.005 --lambda_entropy 0 --window_size 8 --read_message "pointneg"  --message "caser_win8_no_var2"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2  --which_tracker caser --reward_handle "no" --lambda_variance 0.01  --lambda_entropy 0 --window_size 8 --read_message "pointneg"  --message "caser_win8_no_var3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2  --which_tracker caser --reward_handle "no" --lambda_variance 0.05  --lambda_entropy 0 --window_size 8 --read_message "pointneg"  --message "caser_win8_no_var4"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 2  --which_tracker caser --reward_handle "no" --lambda_variance 0.1   --lambda_entropy 0 --window_size 8 --read_message "pointneg"  --message "caser_win8_no_var5"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 5  --which_tracker caser --reward_handle "no" --lambda_variance 0.5   --lambda_entropy 0 --window_size 8 --read_message "pointneg"  --message "caser_win8_no_var6"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 5  --which_tracker caser --reward_handle "no" --lambda_variance 1     --lambda_entropy 0 --window_size 8 --read_message "pointneg"  --message "caser_win8_no_var7"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 5  --which_tracker caser --reward_handle "no" --lambda_variance 5     --lambda_entropy 0 --window_size 8 --read_message "pointneg"  --message "caser_win8_no_var8"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 5  --which_tracker caser --reward_handle "no" --lambda_variance 10    --lambda_entropy 0 --window_size 8 --read_message "pointneg"  --message "caser_win8_no_var9"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 5  --which_tracker caser --reward_handle "no" --lambda_variance 50    --lambda_entropy 0 --window_size 8 --read_message "pointneg"  --message "caser_win8_no_var10"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 5  --which_tracker caser --reward_handle "no" --lambda_variance 100   --lambda_entropy 0 --window_size 8 --read_message "pointneg"  --message "caser_win8_no_var11"  &
#
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 7  --which_tracker caser --reward_handle "no" --lambda_variance 0     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "caser_win5_no_var0"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 7  --which_tracker caser --reward_handle "no" --lambda_variance 0.001 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "caser_win5_no_var1"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 7  --which_tracker caser --reward_handle "no" --lambda_variance 0.005 --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "caser_win5_no_var2"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 7  --which_tracker caser --reward_handle "no" --lambda_variance 0.01  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "caser_win5_no_var3"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 7  --which_tracker caser --reward_handle "no" --lambda_variance 0.05  --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "caser_win5_no_var4"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 7  --which_tracker caser --reward_handle "no" --lambda_variance 0.1   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "caser_win5_no_var5"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4  --which_tracker caser --reward_handle "no" --lambda_variance 0.5   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "caser_win5_no_var6"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4  --which_tracker caser --reward_handle "no" --lambda_variance 1     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "caser_win5_no_var7"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4  --which_tracker caser --reward_handle "no" --lambda_variance 5     --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "caser_win5_no_var8"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4  --which_tracker caser --reward_handle "no" --lambda_variance 10    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "caser_win5_no_var9"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4  --which_tracker caser --reward_handle "no" --lambda_variance 50    --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "caser_win5_no_var10"  &
#python run_Policy_Main.py --env CoatEnv-v0  --cuda 4  --which_tracker caser --reward_handle "no" --lambda_variance 100   --lambda_entropy 0 --window_size 5 --read_message "pointneg"  --message "caser_win5_no_var11"  &


# ==================================================================================================================================

#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0     --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var00"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.001 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var01"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var02"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01  --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var03"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05  --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var04"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1   --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var05"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5   --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var06"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 1     --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var07"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 5     --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var08"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 10    --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var09"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 50    --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var10"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 100   --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var11"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0     --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var00"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.001 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var01"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var02"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01  --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var03"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05  --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var04"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1   --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var05"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5   --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var06"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 1     --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var07"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 5     --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var08"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 10    --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var09"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 50    --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var10"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 100   --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "var11"  &
#
#
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "var04_ent00"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "var04_ent01"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "var04_ent02"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "var04_ent03"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "var04_ent04"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "var04_ent05"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "var04_ent06"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "var04_ent07"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "var04_ent08"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "var04_ent09"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "var04_ent10"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "var04_ent11"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "var04_ent00"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "var04_ent01"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "var04_ent02"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "var04_ent03"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "var04_ent04"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "var04_ent05"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "var04_ent06"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "var04_ent07"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "var04_ent08"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "var04_ent09"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "var04_ent10"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "var04_ent11"  &
#
#
#
#
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "var03_ent00"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "var03_ent01"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "var03_ent02"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "var03_ent03"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "var03_ent04"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "var03_ent05"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "var03_ent06"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "var03_ent07"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "var03_ent08"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "var03_ent09"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "var03_ent10"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "var03_ent11"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "var03_ent00"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "var03_ent01"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "var03_ent02"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "var03_ent03"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "var03_ent04"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "var03_ent05"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "var03_ent06"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "var03_ent07"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "var03_ent08"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "var03_ent09"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "var03_ent10"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "var03_ent11"  &
#
#
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "var05_ent00"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "var05_ent01"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "var05_ent02"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "var05_ent03"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "var05_ent04"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "var05_ent05"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "var05_ent06"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "var05_ent07"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "var05_ent08"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "var05_ent09"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "var05_ent10"  &
#python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "var05_ent11"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "var05_ent00"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "var05_ent01"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "var05_ent02"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "var05_ent03"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "var05_ent04"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "var05_ent05"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "var05_ent06"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "var05_ent07"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "var05_ent08"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "var05_ent09"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "var05_ent10"  &
#python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "var05_ent11"  &

python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "var06_ent00"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "var06_ent01"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "var06_ent02"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "var06_ent03"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "var06_ent04"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "var06_ent05"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "var06_ent06"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "var06_ent07"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "var06_ent08"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "var06_ent09"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "var06_ent10"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "var06_ent11"  &

python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "var06_ent00"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "var06_ent01"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "var06_ent02"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "var06_ent03"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "var06_ent04"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "var06_ent05"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "var06_ent06"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "var06_ent07"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "var06_ent08"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "var06_ent09"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "var06_ent10"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "var06_ent11"  &

python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "var07_ent00"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "var07_ent01"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "var07_ent02"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "var07_ent03"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "var07_ent04"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "var07_ent05"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "var07_ent06"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "var07_ent07"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "var07_ent08"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "var07_ent09"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "var07_ent10"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "var07_ent11"  &

python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "var07_ent00"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "var07_ent01"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "var07_ent02"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "var07_ent03"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "var07_ent04"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "var07_ent05"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "var07_ent06"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "var07_ent07"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "var07_ent08"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "var07_ent09"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "var07_ent10"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "var07_ent11"  &


python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "var08_ent00"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "var08_ent01"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "var08_ent02"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "var08_ent03"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "var08_ent04"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "var08_ent05"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "var08_ent06"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "var08_ent07"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "var08_ent08"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "var08_ent09"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "var08_ent10"  &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "var08_ent11"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "var08_ent00"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "var08_ent01"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "var08_ent02"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "var08_ent03"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "var08_ent04"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "var08_ent05"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "var08_ent06"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "var08_ent07"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "var08_ent08"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "var08_ent09"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "var08_ent10"  &
python run_Policy_Main.py --env KuaiRand-v0  --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "var08_ent11"  &