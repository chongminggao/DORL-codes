
## KuaiEnv

python run_Policy_Main.py --env KuaiEnv-v0  --seed 0 --cuda 4   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "avg_cat_5"    &
python run_Policy_Main.py --env KuaiEnv-v0  --seed 0 --cuda 5   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 5 --read_message "pointneg"  --message "avg_cat"    &
python run_Policy_Main.py --env KuaiEnv-v0  --seed 0 --cuda 6   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat2" --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "avg_cat2"    &
python run_Policy_Main.py --env KuaiEnv-v0  --seed 0 --cuda 7   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "no"   --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "avg_no"    &
python run_Policy_Main.py --env KuaiEnv-v0  --seed 0 --cuda 4   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "mul"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "avg_mul"    &
python run_Policy_Main.py --env KuaiEnv-v0  --seed 0 --cuda 5   --num_leave_compute 4 --leave_threshold 0 --which_tracker "caser"                         --lambda_variance 0.05 --lambda_entropy 5    --window_size 5 --read_message "pointneg"  --message "caser"  &
python run_Policy_Main.py --env KuaiEnv-v0  --seed 0 --cuda 6   --num_leave_compute 4 --leave_threshold 0 --which_tracker "gru"                           --lambda_variance 0.05 --lambda_entropy 5    --window_size 5 --read_message "pointneg"  --message "gru"    &
python run_Policy_Main.py --env KuaiEnv-v0  --seed 0 --cuda 7   --num_leave_compute 4 --leave_threshold 0 --which_tracker "sasrec"                        --lambda_variance 0.05 --lambda_entropy 5    --window_size 5 --read_message "pointneg"  --message "sasrec" &

## KuaiRand

python run_Policy_Main.py --env KuaiRand-v0  --seed 0 --cuda 4   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "avg_cat_5"    &
python run_Policy_Main.py --env KuaiRand-v0  --seed 0 --cuda 5   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 5 --read_message "pointneg"  --message "avg_cat"    &
python run_Policy_Main.py --env KuaiRand-v0  --seed 0 --cuda 6   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat2" --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "avg_cat2"    &
python run_Policy_Main.py --env KuaiRand-v0  --seed 0 --cuda 7   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "no"   --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "avg_no"    &
python run_Policy_Main.py --env KuaiRand-v0  --seed 0 --cuda 4   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "mul"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "avg_mul"    &
python run_Policy_Main.py --env KuaiRand-v0  --seed 0 --cuda 5   --num_leave_compute 4 --leave_threshold 0 --which_tracker "caser"                         --lambda_variance 0.05 --lambda_entropy 5    --window_size 5 --read_message "pointneg"  --message "caser"  &
python run_Policy_Main.py --env KuaiRand-v0  --seed 0 --cuda 6   --num_leave_compute 4 --leave_threshold 0 --which_tracker "gru"                           --lambda_variance 0.05 --lambda_entropy 5    --window_size 5 --read_message "pointneg"  --message "gru"    &
python run_Policy_Main.py --env KuaiRand-v0  --seed 0 --cuda 7   --num_leave_compute 4 --leave_threshold 0 --which_tracker "sasrec"                        --lambda_variance 0.05 --lambda_entropy 5    --window_size 5 --read_message "pointneg"  --message "sasrec" &




python run_Policy_Main.py --env KuaiEnv-v0   --seed 0 --cuda 0   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 1 --read_message "pointneg"  --message "avg_cat_1" &
python run_Policy_Main.py --env KuaiEnv-v0   --seed 0 --cuda 1   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 2 --read_message "pointneg"  --message "avg_cat_2" &
python run_Policy_Main.py --env KuaiEnv-v0   --seed 0 --cuda 2   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "avg_cat_4" &

python run_Policy_Main.py --env KuaiRand-v0  --seed 0 --cuda 0   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 1 --read_message "pointneg"  --message "avg_cat_1" &
python run_Policy_Main.py --env KuaiRand-v0  --seed 0 --cuda 1   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 2 --read_message "pointneg"  --message "avg_cat_2" &
python run_Policy_Main.py --env KuaiRand-v0  --seed 0 --cuda 2   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "avg_cat_4" &


python run_Policy_Main.py --env KuaiEnv-v0    --seed 1 --cuda 4   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "Ours_s1" &
python run_Policy_Main.py --env KuaiEnv-v0    --seed 2 --cuda 5   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "Ours_s2" &
python run_Policy_Main.py --env KuaiEnv-v0    --seed 3 --cuda 6   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "Ours_s3" &
python run_Policy_Main.py --env KuaiEnv-v0    --seed 4 --cuda 7   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "Ours_s4" &
python run_Policy_Main.py --env KuaiRand-v0   --seed 1 --cuda 4   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "Ours_s1" &
python run_Policy_Main.py --env KuaiRand-v0   --seed 2 --cuda 5   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "Ours_s2" &
python run_Policy_Main.py --env KuaiRand-v0   --seed 3 --cuda 6   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "Ours_s3" &
python run_Policy_Main.py --env KuaiRand-v0   --seed 4 --cuda 7   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "Ours_s4" &

python run_Policy_Main.py --env KuaiEnv-v0    --seed 1 --cuda 4   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy  0   --window_size 3 --read_message "pointneg"  --message "MOPO_s1" &
python run_Policy_Main.py --env KuaiEnv-v0    --seed 2 --cuda 5   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy  0   --window_size 3 --read_message "pointneg"  --message "MOPO_s2" &
python run_Policy_Main.py --env KuaiEnv-v0    --seed 3 --cuda 6   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy  0   --window_size 3 --read_message "pointneg"  --message "MOPO_s3" &
python run_Policy_Main.py --env KuaiEnv-v0    --seed 4 --cuda 7   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy  0   --window_size 3 --read_message "pointneg"  --message "MOPO_s4" &
python run_Policy_Main.py --env KuaiRand-v0   --seed 1 --cuda 4   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 0    --window_size 3 --read_message "pointneg"  --message "MOPO_s1" &
python run_Policy_Main.py --env KuaiRand-v0   --seed 2 --cuda 5   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 0    --window_size 3 --read_message "pointneg"  --message "MOPO_s2" &
python run_Policy_Main.py --env KuaiRand-v0   --seed 3 --cuda 6   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 0    --window_size 3 --read_message "pointneg"  --message "MOPO_s3" &
python run_Policy_Main.py --env KuaiRand-v0   --seed 4 --cuda 7   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0.05 --lambda_entropy 0    --window_size 3 --read_message "pointneg"  --message "MOPO_s4" &

python run_Policy_Main.py --env KuaiEnv-v0    --seed 1 --cuda 4   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0 --lambda_entropy  0   --window_size 3 --read_message "pointneg"  --message "MBPO_s1" &
python run_Policy_Main.py --env KuaiEnv-v0    --seed 2 --cuda 5   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0 --lambda_entropy  0   --window_size 3 --read_message "pointneg"  --message "MBPO_s2" &
python run_Policy_Main.py --env KuaiEnv-v0    --seed 3 --cuda 6   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0 --lambda_entropy  0   --window_size 3 --read_message "pointneg"  --message "MBPO_s3" &
python run_Policy_Main.py --env KuaiEnv-v0    --seed 4 --cuda 7   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0 --lambda_entropy  0   --window_size 3 --read_message "pointneg"  --message "MBPO_s4" &
python run_Policy_Main.py --env KuaiRand-v0   --seed 1 --cuda 4   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0 --lambda_entropy 0    --window_size 3 --read_message "pointneg"  --message "MBPO_s1" &
python run_Policy_Main.py --env KuaiRand-v0   --seed 2 --cuda 5   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0 --lambda_entropy 0    --window_size 3 --read_message "pointneg"  --message "MBPO_s2" &
python run_Policy_Main.py --env KuaiRand-v0   --seed 3 --cuda 6   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0 --lambda_entropy 0    --window_size 3 --read_message "pointneg"  --message "MBPO_s3" &
python run_Policy_Main.py --env KuaiRand-v0   --seed 4 --cuda 7   --num_leave_compute 4 --leave_threshold 0 --which_tracker "avg"    --reward_handle "cat"  --lambda_variance 0 --lambda_entropy 0    --window_size 3 --read_message "pointneg"  --message "MBPO_s4" &



