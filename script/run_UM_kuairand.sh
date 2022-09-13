

#python 1.run_worldModel_kuairand.py --tau 0 --message "1"  --cuda 0 &
#python 1.run_worldModel_kuairand.py --tau 0 --message "2"  --cuda 1 --no_use_pairwise &
#python 1.run_worldModel_kuairand.py --tau 0 --message "3"  --cuda 2 --feature_dim 4 &
#python 1.run_worldModel_kuairand.py --tau 0 --message "9"  --cuda 3 --feature_dim 16 &
#python 1.run_worldModel_kuairand.py --tau 10 --message "4"  --cuda 3 --feature_dim 16 &
#python 1.run_worldModel_kuairand.py --tau 100 --message "5"  --cuda 0  &
#python 1.run_worldModel_kuairand.py --tau 1000 --message "6"  --cuda 1 &
#python 1.run_worldModel_kuairand.py --tau 10000 --message "7"  --cuda 2 &
#python 1.run_worldModel_kuairand.py --tau 0 --message "8"  --cuda 3 --epoch 10 &


python run_worldModel_kuairand.py --tau 0     --message "like1" --yfeat "is_like"  --no_use_pairwise --cuda 1 &
python run_worldModel_kuairand.py --tau 0     --message "like2" --yfeat "is_like"  --no_use_pairwise --cuda 2 --feature_dim 4 &
python run_worldModel_kuairand.py --tau 0     --message "like3" --yfeat "is_like"  --no_use_pairwise --cuda 3 --feature_dim 16 &
python run_worldModel_kuairand.py --tau 10    --message "like4" --yfeat "is_like"  --no_use_pairwise --cuda 0 &
python run_worldModel_kuairand.py --tau 100   --message "like5" --yfeat "is_like"  --no_use_pairwise --cuda 0 &
python run_worldModel_kuairand.py --tau 1000  --message "like6" --yfeat "is_like"  --no_use_pairwise --cuda 1 &
python run_worldModel_kuairand.py --tau 10000 --message "like7" --yfeat "is_like"  --no_use_pairwise --cuda 2 &
python run_worldModel_kuairand.py --tau 0     --message "like8" --yfeat "is_like"  --no_use_pairwise --cuda 3 --epoch 10 &

python run_worldModel_kuairand.py --tau 0     --message "click1" --yfeat "is_click"  --no_use_pairwise --cuda 1 &
python run_worldModel_kuairand.py --tau 0     --message "click2" --yfeat "is_click"  --no_use_pairwise --cuda 2 --feature_dim 4 &
python run_worldModel_kuairand.py --tau 0     --message "click3" --yfeat "is_click"  --no_use_pairwise --cuda 3 --feature_dim 16 &
python run_worldModel_kuairand.py --tau 10    --message "click4" --yfeat "is_click"  --no_use_pairwise --cuda 0 &
python run_worldModel_kuairand.py --tau 100   --message "click5" --yfeat "is_click"  --no_use_pairwise --cuda 0 &
python run_worldModel_kuairand.py --tau 1000  --message "click6" --yfeat "is_click"  --no_use_pairwise --cuda 1 &
python run_worldModel_kuairand.py --tau 10000 --message "click7" --yfeat "is_click"  --no_use_pairwise --cuda 2 &
python run_worldModel_kuairand.py --tau 0     --message "click8" --yfeat "is_click"  --no_use_pairwise --cuda 3 --epoch 10 &

python run_worldModel_kuairand.py --tau 0     --message "view1" --yfeat "long_view"  --no_use_pairwise --cuda 1 &
python run_worldModel_kuairand.py --tau 0     --message "view2" --yfeat "long_view"  --no_use_pairwise --cuda 2 --feature_dim 4 &
python run_worldModel_kuairand.py --tau 0     --message "view3" --yfeat "long_view"  --no_use_pairwise --cuda 3 --feature_dim 16 &
python run_worldModel_kuairand.py --tau 10    --message "view4" --yfeat "long_view"  --no_use_pairwise --cuda 0 &
python run_worldModel_kuairand.py --tau 100   --message "view5" --yfeat "long_view"  --no_use_pairwise --cuda 0 &
python run_worldModel_kuairand.py --tau 1000  --message "view6" --yfeat "long_view"  --no_use_pairwise --cuda 1 &
python run_worldModel_kuairand.py --tau 10000 --message "view7" --yfeat "long_view"  --no_use_pairwise --cuda 2 &
python run_worldModel_kuairand.py --tau 0     --message "view8" --yfeat "long_view"  --no_use_pairwise --cuda 3 --epoch 10 &

python run_worldModel_kuairand.py --tau 0     --message "hybrid1" --yfeat "hybrid"  --no_use_pairwise --cuda 1 &
python run_worldModel_kuairand.py --tau 0     --message "hybrid2" --yfeat "hybrid"  --no_use_pairwise --cuda 2 --feature_dim 4 &
python run_worldModel_kuairand.py --tau 0     --message "hybrid3" --yfeat "hybrid"  --no_use_pairwise --cuda 3 --feature_dim 16 &
python run_worldModel_kuairand.py --tau 10    --message "hybrid4" --yfeat "hybrid"  --no_use_pairwise --cuda 0 &
python run_worldModel_kuairand.py --tau 100   --message "hybrid5" --yfeat "hybrid"  --no_use_pairwise --cuda 0 &
python run_worldModel_kuairand.py --tau 1000  --message "hybrid6" --yfeat "hybrid"  --no_use_pairwise --cuda 1 &
python run_worldModel_kuairand.py --tau 10000 --message "hybrid7" --yfeat "hybrid"  --no_use_pairwise --cuda 2 &
python run_worldModel_kuairand.py --tau 0     --message "hybrid8" --yfeat "hybrid"  --no_use_pairwise --cuda 3 --epoch 10 &


python run_worldModel_kuairand.py --tau 0   --message "sgd all 0.1 hybrid" --yfeat "hybrid"  --no_use_pairwise --cuda 3 &
python run_worldModel_kuairand.py --tau 0   --message "adam all hybrid" --yfeat "hybrid"  --no_use_pairwise --cuda 3 &
python run_worldModel_kuairand.py --tau 0   --message "sgd all 0.05 hybrid" --yfeat "hybrid"  --no_use_pairwise --cuda 3 &
python run_worldModel_kuairand.py --tau 0   --message "sgd all 0.01 hybrid" --yfeat "hybrid"  --no_use_pairwise --cuda 3 &

python run_worldModel_kuairand.py --tau 0   --message "sgd all 0.1 hybrid" --yfeat "hybrid"  --no_use_pairwise --cuda 3 &
python run_worldModel_kuairand.py --tau 0   --message "adam all hybrid" --yfeat "hybrid"  --no_use_pairwise --cuda 3 &
python run_worldModel_kuairand.py --tau 0   --message "sgd all 0.05 hybrid" --yfeat "hybrid"  --no_use_pairwise --cuda 3 &
python run_worldModel_kuairand.py --tau 0   --message "sgd all 0.01 hybrid" --yfeat "hybrid"  --no_use_pairwise --cuda 3 &


python run_worldModel_kuairand.py --tau 0   --message "sgd 0.1 long_view" --yfeat "long_view"  --no_use_pairwise --cuda 3 &