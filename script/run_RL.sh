

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

python run_A2CPolicy_coat2.py --cuda 7  --read_message "pointneg" --window 1 --message "freeze_v2win1" --is_freeze_emb &
python run_A2CPolicy_coat2.py --cuda 7  --read_message "pointneg" --window 2 --message "freeze_v2win2" --is_freeze_emb &
python run_A2CPolicy_coat2.py --cuda 7  --read_message "pointneg" --window 1 --message "freeze_v2win1_user" --is_use_userEmbedding --is_freeze_emb &
python run_A2CPolicy_coat2.py --cuda 7  --read_message "pointneg" --window 2 --message "freeze_v2win2_user" --is_use_userEmbedding --is_freeze_emb &