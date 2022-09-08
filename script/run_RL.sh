#python run_A2CPolicy.py --read_message "UM tau1000" --message "A2C tau100" --cuda 1 --leave_threshold 1 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau0" --message "A2C tau0" --cuda 2 --leave_threshold 1 --num_leave_compute 5 &
#
#python run_ppo_withEmbedding.py --read_message "UM tau1000" --message "PPOemb tau1000" --cuda 2 --leave_threshold 1 --num_leave_compute 5 &
#python run_ppo_withEmbedding.py --read_message "UM tau0" --message "PPOemb tau0" --cuda 1 --leave_threshold 1 --num_leave_compute 5 &
#
#python run_PPOPolicy.py --read_message "UM tau1000" --message "PPO tau1000" --cuda 1 --leave_threshold 1 --num_leave_compute 5 &
#python run_PPOPolicy.py --read_message "UM tau0" --message "PPO tau0" --cuda 2 --leave_threshold 1 --num_leave_compute 5 &


#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1 --message "A2C t1" --cuda 1 --leave_threshold 1 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 10 --message "A2C t10" --cuda 2 --leave_threshold 1 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "A2C t1000" --cuda 1 --leave_threshold 1 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 10000 --message "A2C t10000" --cuda 2 --leave_threshold 1 --num_leave_compute 5 &





# dm4
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1 --message "A2C win2" --window 2 --cuda 1 --leave_threshold 1 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 10 --message "A2C win3" --window 3 --cuda 2 --leave_threshold 1 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "A2C win5" --window 5 --cuda 1 --leave_threshold 1 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 10000 --message "A2C win8" --window 8 --cuda 2 --leave_threshold 1 --num_leave_compute 5 &
#
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1 --message "A2C t1 leave2" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 10 --message "A2C t10 leave2" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "A2C t1000 leave2" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 10000 --message "A2C t10000 leave2" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1 --message "A2C t1 leave3" --cuda 1 --leave_threshold 3 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 10 --message "A2C t10 leave3" --cuda 2 --leave_threshold 3 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "A2C t1000 leave3" --cuda 1 --leave_threshold 3 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 10000 --message "A2C t10000 leave3" --cuda 2 --leave_threshold 3 --num_leave_compute 5 &

# dm4
#python3 run_A2CPolicy.py --read_message "UM tau1000" --tau 1 --message "AC 1" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --read_message "UM tau1000" --tau 10 --message "AC 2" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --read_message "UM tau1000" --tau 100 --message "AC 3" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "AC 4" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --read_message "UM tau1000" --tau 10000 --message "AC 5" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#

#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "A2C win2" --window 2 --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "A2C win3" --window 3 --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "A2C win5" --window 5 --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#
## dm3
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "A2C win8" --window 8 --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "A2C win2 noUser" --window 2 --cuda 2 --no_use_userEmbedding --leave_threshold 2 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "A2C win3 noUser" --window 3 --cuda 0 --no_use_userEmbedding --leave_threshold 2 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "A2C win5 noUser" --window 5 --cuda 2 --no_use_userEmbedding --leave_threshold 2 --num_leave_compute 5 &
#python run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "A2C win8 noUser" --window 8 --cuda 0 --no_use_userEmbedding --leave_threshold 2 --num_leave_compute 5 &

#/share/gaochongming/miniconda3/envs/DRL2/bin/python3 run_ppo_withEmbedding.py --read_message "UM tau1000" --tau 1 --message "PPOemb t1 leave3" --cuda 1 --leave_threshold 3 --num_leave_compute 5 &
#/share/gaochongming/miniconda3/envs/DRL2/bin/python3 run_ppo_withEmbedding.py --read_message "UM tau1000" --tau 10 --message "PPOemb t10 leave3" --cuda 1 --leave_threshold 3 --num_leave_compute 5 &
#/share/gaochongming/miniconda3/envs/DRL2/bin/python3 run_ppo_withEmbedding.py --read_message "UM tau1000" --tau 1000 --message "PPOemb t1000 leave3" --cuda 1 --leave_threshold 3 --num_leave_compute 5 &
#/share/gaochongming/miniconda3/envs/DRL2/bin/python3 run_ppo_withEmbedding.py --read_message "UM tau1000" --tau 10000 --message "PPOemb t10000 leave3" --cuda 1 --leave_threshold 3 --num_leave_compute 5 &


#python3 run_A2CPolicy.py --read_message "UM tau1000" --tau 1 --message "AC 1" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --read_message "UM tau1000" --tau 10 --message "AC 2" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --read_message "UM tau1000" --tau 100 --message "AC 3" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --read_message "UM tau1000" --tau 1000 --message "AC 4" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --read_message "UM tau1000" --tau 10000 --message "AC 5" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &


#python3 run_A2CPolicy.py --eps  0 --read_message   "UM tau1000" --tau 10 --message "AC eps0 no" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .025 --read_message "UM tau1000" --tau 10 --message "AC eps1 no" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .05 --read_message  "UM tau1000" --tau 10 --message "AC eps2 no" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .075 --read_message "UM tau1000" --tau 10 --message "AC eps3 no" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .1 --read_message   "UM tau1000" --tau 10 --message "AC eps4 no" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#
#python3 run_A2CPolicy.py --eps  0 --read_message   "UM tau1000" --tau 10 --message "AC eps0 no" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .025 --read_message "UM tau1000" --tau 10 --message "AC eps1 no" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .05 --read_message  "UM tau1000" --tau 10 --message "AC eps2 no" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .075 --read_message "UM tau1000" --tau 10 --message "AC eps3 no" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .1 --read_message   "UM tau1000" --tau 10 --message "AC eps4 no" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &

#python3 run_A2CPolicy.py --eps  0 --read_message "UM tau1000" --tau 10 --message "AC eps0" --cuda 0 --leave_threshold 2 --num_leave_compute 5 --is_use_userEmbedding &
#python3 run_A2CPolicy.py --eps .1 --read_message "UM tau1000" --tau 10 --message "AC eps1" --cuda 0 --leave_threshold 2 --num_leave_compute 5 --is_use_userEmbedding &
#python3 run_A2CPolicy.py --eps .2 --read_message "UM tau1000" --tau 10 --message "AC eps2" --cuda 0 --leave_threshold 2 --num_leave_compute 5 --is_use_userEmbedding &
#python3 run_A2CPolicy.py --eps .3 --read_message "UM tau1000" --tau 10 --message "AC eps3" --cuda 1 --leave_threshold 2 --num_leave_compute 5 --is_use_userEmbedding &
#python3 run_A2CPolicy.py --eps .4 --read_message "UM tau1000" --tau 10 --message "AC eps4" --cuda 1 --leave_threshold 2 --num_leave_compute 5 --is_use_userEmbedding &
#python3 run_A2CPolicy.py --eps .5 --read_message "UM tau1000" --tau 10 --message "AC eps5" --cuda 1 --leave_threshold 2 --num_leave_compute 5 --is_use_userEmbedding &




#python3 run_A2CPolicy.py --eps  0 --read_message   "UM tau1000" --tau 10 --window 1 --message "eps1" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .025 --read_message "UM tau1000" --tau 10 --window 1 --message "eps2" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .05 --read_message  "UM tau1000" --tau 10 --window 1 --message "eps3" --cuda 3 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .075 --read_message "UM tau1000" --tau 10 --window 1 --message "eps4" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .1 --read_message   "UM tau1000" --tau 10 --window 1 --message "eps5" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#
python3 run_A2CPolicy.py --eps  0 --read_message   "UM tau1000" --tau 10 --window 2 --message "eps6" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &
python3 run_A2CPolicy.py --eps .025 --read_message "UM tau1000" --tau 10 --window 2 --message "eps7" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
python3 run_A2CPolicy.py --eps .05 --read_message  "UM tau1000" --tau 10 --window 2 --message "eps8" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
python3 run_A2CPolicy.py --eps .075 --read_message "UM tau1000" --tau 10 --window 2 --message "eps9" --cuda 3 --leave_threshold 2 --num_leave_compute 5 &
python3 run_A2CPolicy.py --eps .1 --read_message   "UM tau1000" --tau 10 --window 2 --message "eps10" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &


python3 run_A2CPolicy.py --eps  0 --read_message   "UM tau1000" --tau 10 --window 3 --message "eps11" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
python3 run_A2CPolicy.py --eps .025 --read_message "UM tau1000" --tau 10 --window 3 --message "eps12" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
python3 run_A2CPolicy.py --eps .05 --read_message  "UM tau1000" --tau 10 --window 3 --message "eps13" --cuda 3 --leave_threshold 2 --num_leave_compute 5 &
python3 run_A2CPolicy.py --eps .075 --read_message "UM tau1000" --tau 10 --window 3 --message "eps14" --cuda 0 --leave_threshold 2 --num_leave_compute 5 &

python3 run_A2CPolicy.py --eps .1 --read_message   "UM tau1000" --tau 10 --window 3 --message "eps15" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#
#
#python3 run_A2CPolicy.py --eps  0 --read_message   "UM tau1000" --tau 10 --window 5 --message "eps16" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .025 --read_message "UM tau1000" --tau 10 --window 5 --message "eps17" --cuda 3 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .05 --read_message  "UM tau1000" --tau 10 --window 5 --message "eps18" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .075 --read_message "UM tau1000" --tau 10 --window 5 --message "eps19" --cuda 3 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .1 --read_message   "UM tau1000" --tau 10 --window 5 --message "eps20" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &

#python3 run_A2CPolicy.py --eps  0 --read_message   "UM tau1000" --tau 10 --window 10 --message "eps21" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .025 --read_message "UM tau1000" --tau 10 --window 10 --message "eps22" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .05 --read_message  "UM tau1000" --tau 10 --window 10 --message "eps23" --cuda 3 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .075 --read_message "UM tau1000" --tau 10 --window 10 --message "eps24" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .1 --read_message   "UM tau1000" --tau 10 --window 10 --message "eps25" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#
#python3 run_A2CPolicy.py --eps  0 --read_message   "UM tau1000" --tau 10 --window 15 --message "eps26" --cuda 3 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .025 --read_message "UM tau1000" --tau 10 --window 15 --message "eps27" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .05 --read_message  "UM tau1000" --tau 10 --window 15 --message "eps28" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .075 --read_message "UM tau1000" --tau 10 --window 15 --message "eps29" --cuda 3 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps .1 --read_message   "UM tau1000" --tau 10 --window 15 --message "eps30" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &

#python3 run_A2CPolicy.py --eps  0.1 --read_message "UM tau1000" --tau 10 --window 3 --message "eps31" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps  0.2 --read_message "UM tau1000" --tau 10 --window 3 --message "eps32" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps  0.3 --read_message "UM tau1000" --tau 10 --window 3 --message "eps33" --cuda 1 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps  0.4 --read_message "UM tau1000" --tau 10 --window 3 --message "eps34" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps  0.5 --read_message "UM tau1000" --tau 10 --window 3 --message "eps35" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps  0.6 --read_message "UM tau1000" --tau 10 --window 3 --message "eps36" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps  0.7 --read_message "UM tau1000" --tau 10 --window 3 --message "eps37" --cuda 2 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps  0.8 --read_message "UM tau1000" --tau 10 --window 3 --message "eps38" --cuda 3 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps  0.9 --read_message "UM tau1000" --tau 10 --window 3 --message "eps39" --cuda 3 --leave_threshold 2 --num_leave_compute 5 &
#python3 run_A2CPolicy.py --eps  1   --read_message "UM tau1000" --tau 10 --window 3 --message "eps40" --cuda 3 --leave_threshold 2 --num_leave_compute 5 &



