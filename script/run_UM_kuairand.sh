

#python 1.run_worldModel_kuairand.py --tau 0 --message "1"  --cuda 0 &
#python 1.run_worldModel_kuairand.py --tau 0 --message "2"  --cuda 1 --no_use_pairwise &
#python 1.run_worldModel_kuairand.py --tau 0 --message "3"  --cuda 2 --feature_dim 4 &
#python 1.run_worldModel_kuairand.py --tau 0 --message "9"  --cuda 3 --feature_dim 16 &
python 1.run_worldModel_kuairand.py --tau 10 --message "4"  --cuda 3 --feature_dim 16 &
python 1.run_worldModel_kuairand.py --tau 100 --message "5"  --cuda 0  &
python 1.run_worldModel_kuairand.py --tau 1000 --message "6"  --cuda 1 &
python 1.run_worldModel_kuairand.py --tau 10000 --message "7"  --cuda 2 &
#python 1.run_worldModel_kuairand.py --tau 0 --message "8"  --cuda 3 --epoch 10 &

