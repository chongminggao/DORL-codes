python run_A2CPolicy.py --read_message "UM tau1000" --message "A2C tau100" --cuda 1 --leave_threshold 1 --num_leave_compute 5 &
python run_A2CPolicy.py --read_message "UM tau0" --message "A2C tau0" --cuda 2 --leave_threshold 1 --num_leave_compute 5 &

python run_ppo_withEmbedding.py --read_message "UM tau1000" --message "PPOemb tau1000" --cuda 2 --leave_threshold 1 --num_leave_compute 5 &
python run_ppo_withEmbedding.py --read_message "UM tau0" --message "PPOemb tau0" --cuda 1 --leave_threshold 1 --num_leave_compute 5 &

python run_PPOPolicy.py --read_message "UM tau1000" --message "PPO tau1000" --cuda 1 --leave_threshold 1 --num_leave_compute 5 &
python run_PPOPolicy.py --read_message "UM tau0" --message "PPO tau0" --cuda 2 --leave_threshold 1 --num_leave_compute 5 &