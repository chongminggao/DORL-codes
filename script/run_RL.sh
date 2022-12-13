

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
#python run_Policy_BCQ.py  --read_message "pointneg" --cuda 1 --env CoatEnv-v0  --message "CoatEnv"  &
#python run_Policy_BCQ.py  --read_message "pointneg" --cuda 2 --env YahooEnv-v0 --message "YahooEnv" &
#python run_Policy_BCQ.py  --read_message "pointneg" --cuda 4 --env KuaiRand-v0 --message "KuaiRand" &
python run_Policy_BCQ.py  --read_message "pointneg" --cuda 1 --env KuaiEnv-v0  --message "KuaiRand" &
#
#python run_Policy_CQL.py  --read_message "pointneg" --cuda 1 --env CoatEnv-v0  --message "CoatEnv"  &
#python run_Policy_CQL.py  --read_message "pointneg" --cuda 2 --env YahooEnv-v0 --message "YahooEnv" &
#python run_Policy_CQL.py  --read_message "pointneg" --cuda 4 --env KuaiRand-v0 --message "KuaiRand" &
python run_Policy_CQL.py  --read_message "pointneg" --cuda 1 --env KuaiEnv-v0  --message "KuaiRand" &
#
#python run_Policy_CRR.py  --read_message "pointneg" --cuda 4 --env CoatEnv-v0  --message "CoatEnv"  &
#python run_Policy_CRR.py  --read_message "pointneg" --cuda 1 --env YahooEnv-v0 --message "YahooEnv" &
#python run_Policy_CRR.py  --read_message "pointneg" --cuda 6 --env KuaiRand-v0 --message "KuaiRand" &
python run_Policy_CRR.py  --read_message "pointneg" --cuda 7 --env KuaiEnv-v0  --message "KuaiRand" &


#python run_Policy_Main.py --read_message "pointneg" --cuda 1 --env CoatEnv-v0  --message "CoatEnv"  &
#python run_Policy_Main.py --read_message "pointneg" --cuda 2 --env YahooEnv-v0 --message "YahooEnv" &
#python run_Policy_Main.py --read_message "pointneg" --cuda 6 --env KuaiRand-v0 --message "KuaiRand" &
#python run_Policy_Main.py --read_message "pointneg" --cuda 7 --env KuaiEnv-v0  --message "KuaiRand" &

#python run_Policy_SQN.py  --cuda 1 --which_tracker caser --which_head shead --env CoatEnv-v0  --message "caser_shead" &
#python run_Policy_SQN.py  --cuda 2 --which_tracker caser --which_head shead --env YahooEnv-v0 --message "caser_shead" &
#python run_Policy_SQN.py  --cuda 1 --which_tracker caser --which_head shead --env KuaiRand-v0 --message "caser_shead" &
#python run_Policy_SQN.py  --cuda 2 --which_tracker caser --which_head shead --env KuaiEnv-v0  --message "caser_shead" &
#
#python run_Policy_SQN.py  --cuda 1 --which_tracker caser --which_head qhead --env CoatEnv-v0  --message "caser_qhead" &
#python run_Policy_SQN.py  --cuda 2 --which_tracker caser --which_head qhead --env YahooEnv-v0 --message "caser_qhead" &
#python run_Policy_SQN.py  --cuda 1 --which_tracker caser --which_head qhead --env KuaiRand-v0 --message "caser_qhead" &
python run_Policy_SQN.py  --cuda 2 --which_tracker caser --which_head qhead --env KuaiEnv-v0  --message "caser_qhead" &


#python run_Policy_SQN.py  --cuda 4 --which_tracker gru --which_head shead --env CoatEnv-v0  --message "gru_shead" &
#python run_Policy_SQN.py  --cuda 5 --which_tracker gru --which_head shead --env YahooEnv-v0 --message "gru_shead" &
#python run_Policy_SQN.py  --cuda 4 --which_tracker gru --which_head shead --env KuaiRand-v0 --message "gru_shead" &
#python run_Policy_SQN.py  --cuda 5 --which_tracker gru --which_head shead --env KuaiEnv-v0  --message "gru_shead" &
#
#python run_Policy_SQN.py  --cuda 3 --which_tracker sasrec --which_head qhead --env CoatEnv-v0  --message "sasrec_qhead" &
#python run_Policy_SQN.py  --cuda 6 --which_tracker sasrec --which_head qhead --env YahooEnv-v0 --message "sasrec_qhead" &
#python run_Policy_SQN.py  --cuda 3 --which_tracker sasrec --which_head qhead --env KuaiRand-v0 --message "sasrec_qhead" &
python run_Policy_SQN.py  --cuda 7 --which_tracker sasrec --which_head qhead --env KuaiEnv-v0  --message "sasrec_qhead" &
#
#
#python run_Policy_SQN.py  --cuda 6 --which_tracker gru --which_head qhead --env CoatEnv-v0  --message "gru_qhead" &
#python run_Policy_SQN.py  --cuda 7 --which_tracker gru --which_head qhead --env YahooEnv-v0 --message "gru_qhead" &
#python run_Policy_SQN.py  --cuda 4 --which_tracker gru --which_head qhead --env KuaiRand-v0 --message "gru_qhead" &
python run_Policy_SQN.py  --cuda 2 --which_tracker gru --which_head qhead --env KuaiEnv-v0  --message "gru_qhead" &
##
#python run_Policy_SQN.py  --cuda 6 --which_tracker gru --which_head bcq --env CoatEnv-v0  --message "gru_bcq" &
#python run_Policy_SQN.py  --cuda 7 --which_tracker gru --which_head bcq --env YahooEnv-v0 --message "gru_bcq" &
#python run_Policy_SQN.py  --cuda 4 --which_tracker gru --which_head bcq --env KuaiRand-v0 --message "gru_bcq" &
#python run_Policy_SQN.py  --cuda 5 --which_tracker gru --which_head bcq --env KuaiEnv-v0  --message "gru_bcq" &