#python run_worldModel.py --tau 0 --message "UM tau0"  --cuda 0 &
#python run_worldModel.py --tau 10 --message "UM tau10" --cuda 1 &
#python run_worldModel.py --tau 100 --message "UM tau100" --cuda 0 &
#python run_worldModel.py --tau 1000 --message "UM tau1000" --cuda 1 &
#python run_worldModel.py --tau 10000 --message "UM tau10000" --cuda 0 &

python run_worldModel.py --tau 0 --message "pointwise tau0"  --cuda 1 --no_use_pairwise &
python run_worldModel.py --tau 10 --message "pointwise tau10" --cuda 2 --no_use_pairwise &
python run_worldModel.py --tau 100 --message "pointwise tau100" --cuda 3 --no_use_pairwise &
python run_worldModel.py --tau 1000 --message "pointwise tau1000" --cuda 1 --no_use_pairwise &
python run_worldModel.py --tau 10000 --message "pointwise tau10000" --cuda 2 --no_use_pairwise &