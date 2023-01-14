
# ==================================================================================================================================

sleep 10
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0     --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "evar00" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.001 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "evar01" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "evar02" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01  --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "evar03" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05  --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "evar04" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1   --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "evar05" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5   --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "evar06" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 1     --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "evar07" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 5     --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "evar08" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 10    --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "evar09" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 50    --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "evar10" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 100   --lambda_entropy 0 --window_size 3 --read_message "pointneg"  --message "evar11" &

python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "evar02_ent00" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "evar02_ent01" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "evar02_ent02" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "evar02_ent03" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "evar02_ent04" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "evar02_ent05" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "evar02_ent06" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "evar02_ent07" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "evar02_ent08" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "evar02_ent09" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "evar02_ent10" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.005 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "evar02_ent11" &


python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "evar03_ent00" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "evar03_ent01" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "evar03_ent02" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "evar03_ent03" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "evar03_ent04" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "evar03_ent05" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "evar03_ent06" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "evar03_ent07" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "evar03_ent08" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "evar03_ent09" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "evar03_ent10" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "evar03_ent11" &

python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "evar04_ent00" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "evar04_ent01" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "evar04_ent02" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "evar04_ent03" &




sleep 7200

python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "evar04_ent04" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "evar04_ent05" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "evar04_ent06" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "evar04_ent07" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "evar04_ent08" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "evar04_ent09" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "evar04_ent10" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "evar04_ent11" &


python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "evar05_ent00" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "evar05_ent01" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "evar05_ent02" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "evar05_ent03" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "evar05_ent04" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "evar05_ent05" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "evar05_ent06" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "evar05_ent07" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "evar05_ent08" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "evar05_ent09" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "evar05_ent10" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 0.1 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "evar05_ent11" &

python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "evar06_ent00" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "evar06_ent01" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "evar06_ent02" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "evar06_ent03" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "evar06_ent04" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "evar06_ent05" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "evar06_ent06" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "evar06_ent07" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5 --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "evar06_ent08" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5 --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "evar06_ent09" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6 --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "evar06_ent10" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6 --which_tracker avg --reward_handle "cat" --lambda_variance 0.5 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "evar06_ent11" &

python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "evar07_ent00" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "evar07_ent01" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "evar07_ent02" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "evar07_ent03" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "evar07_ent04" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "evar07_ent05" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "evar07_ent06" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "evar07_ent07" &
#
sleep 7200
#
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "evar07_ent08" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "evar07_ent09" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "evar07_ent10" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 1 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "evar07_ent11" &

python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "evar08_ent00" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "evar08_ent01" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "evar08_ent02" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "evar08_ent03" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "evar08_ent04" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "evar08_ent05" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "evar08_ent06" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "evar08_ent07" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "evar08_ent08" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "evar08_ent09" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "evar08_ent10" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 5 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "evar08_ent11" &

python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 10 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "evar09_ent00" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 10 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "evar09_ent01" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 10 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "evar09_ent02" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 10 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "evar09_ent03" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 10 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "evar09_ent04" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 10 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "evar09_ent05" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 10 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "evar09_ent06" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 10 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "evar09_ent07" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 10 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "evar09_ent08" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 10 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "evar09_ent09" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 10 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "evar09_ent10" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 10 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "evar09_ent11" &

python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 50 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "evar10_ent00" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 50 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "evar10_ent01" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 50 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "evar10_ent02" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 50 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "evar10_ent03" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 50 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "evar10_ent04" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 50 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "evar10_ent05" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 50 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "evar10_ent06" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 50 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "evar10_ent07" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 50 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "evar10_ent08" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 50 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "evar10_ent09" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 50 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "evar10_ent10" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 50 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "evar10_ent11" &
#

sleep 7200

python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 100 --lambda_entropy 0     --window_size 3 --read_message "pointneg"  --message "evar11_ent00" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 100 --lambda_entropy 0.001 --window_size 3 --read_message "pointneg"  --message "evar11_ent01" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 100 --lambda_entropy 0.005 --window_size 3 --read_message "pointneg"  --message "evar11_ent02" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 100 --lambda_entropy 0.01  --window_size 3 --read_message "pointneg"  --message "evar11_ent03" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 4  --which_tracker avg --reward_handle "cat" --lambda_variance 100 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "evar11_ent04" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --lambda_variance 100 --lambda_entropy 0.1   --window_size 3 --read_message "pointneg"  --message "evar11_ent05" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 6  --which_tracker avg --reward_handle "cat" --lambda_variance 100 --lambda_entropy 0.5   --window_size 3 --read_message "pointneg"  --message "evar11_ent06" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 7  --which_tracker avg --reward_handle "cat" --lambda_variance 100 --lambda_entropy 1     --window_size 3 --read_message "pointneg"  --message "evar11_ent07" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 0  --which_tracker avg --reward_handle "cat" --lambda_variance 100 --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "evar11_ent08" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 1  --which_tracker avg --reward_handle "cat" --lambda_variance 100 --lambda_entropy 10    --window_size 3 --read_message "pointneg"  --message "evar11_ent09" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 2  --which_tracker avg --reward_handle "cat" --lambda_variance 100 --lambda_entropy 50    --window_size 3 --read_message "pointneg"  --message "evar11_ent10" &
python run_Policy_Main.py --env KuaiEnv-v0   --cuda 3  --which_tracker avg --reward_handle "cat" --lambda_variance 100 --lambda_entropy 100   --window_size 3 --read_message "pointneg"  --message "evar11_ent11" &