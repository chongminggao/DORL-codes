 # lab9 ================================
python run_worldModel_coat.py --tau 0  --cuda 1  --message "point"  --loss "point" &
python run_worldModel_coat.py --tau 0  --cuda 2  --message "pair"  --loss "pair" &
python run_worldModel_coat.py --tau 0  --cuda 3  --message "pp"  --loss "pp" &
python run_worldModel_coat.py --tau 0  --cuda 4  --message "pointneg"  --loss "pointneg" &

python run_worldModel_yahoo.py --tau 0 --cuda 5   --message "point"  --loss "point" &
python run_worldModel_yahoo.py --tau 0 --cuda 6   --message "pair"  --loss "pair" &
python run_worldModel_yahoo.py --tau 0 --cuda 1   --message "pp"  --loss "pp" &
python run_worldModel_yahoo.py --tau 0 --cuda 2   --message "pointneg"  --loss "pointneg" &

 # lab9 ================================
python run_worldModel_kuairand.py --tau 0  --cuda 3   --message "point"  --loss "point" &
python run_worldModel_kuairand.py --tau 0  --cuda 4   --message "pair"  --loss "pair" &
python run_worldModel_kuairand.py --tau 0  --cuda 5   --message "pp"  --loss "pp" &
python run_worldModel_kuairand.py --tau 0  --cuda 6   --message "pointneg"  --loss "pointneg" &

python run_worldModel_kuairec.py  --tau 0  --cuda 1   --message "point"  --loss "point" &
python run_worldModel_kuairec.py  --tau 0  --cuda 2   --message "pair"  --loss "pair" &
python run_worldModel_kuairec.py  --tau 0  --cuda 3   --message "pp"  --loss "pp" &
python run_worldModel_kuairec.py  --tau 0  --cuda 4   --message "pointneg"  --loss "pointneg" &









# test on lab5
#python run_worldModel_kuairand.py --tau 0  --cuda 0  --batch_size 64 --feature_dim 4  --entity_dim  4 --message "64-4"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 1  --batch_size 128 --feature_dim 4  --entity_dim  4 --message "128-4"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 2  --batch_size 256 --feature_dim 4  --entity_dim  4 --message "256-4"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 3  --batch_size 512 --feature_dim 4  --entity_dim  4 --message "512-4"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 4  --batch_size 1024 --feature_dim 4 --entity_dim  4  --message "1024-4"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 5  --batch_size 2048 --feature_dim 4 --entity_dim  4  --message "2048-4"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 6  --batch_size 4096 --feature_dim 4 --entity_dim  4  --message "4096-4"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 6  --batch_size 10000 --feature_dim 4 --entity_dim 4   --message "10000-4"  --loss "pointneg" &
#
#python run_worldModel_kuairand.py --tau 0  --cuda 0  --batch_size 64 --feature_dim 8  --entity_dim  8 --message "64-8"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 1  --batch_size 128 --feature_dim 8  --entity_dim 8 --message "128-8"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 2  --batch_size 256 --feature_dim 8  --entity_dim 8 --message "256-8"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 3  --batch_size 512 --feature_dim 8  --entity_dim 8 --message "512-8"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 4  --batch_size 1024 --feature_dim 8 --entity_dim 8  --message "1024-8"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 5  --batch_size 2048 --feature_dim 8 --entity_dim 8  --message "2048-8"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 6  --batch_size 4096 --feature_dim 8 --entity_dim 8  --message "4096-8"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 6  --batch_size 10000 --feature_dim 8 --entity_dim 8  --message "10000-8"  --loss "pointneg" &
#
#python run_worldModel_kuairand.py --tau 0  --cuda 0  --batch_size 64    --feature_dim 16 --entity_dim  16 --message "64-16"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 1  --batch_size 128   --feature_dim 16 --entity_dim  16  --message "128-16"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 2  --batch_size 256   --feature_dim 16 --entity_dim  16  --message "256-16"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 3  --batch_size 512   --feature_dim 16 --entity_dim  16  --message "512-16"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 4  --batch_size 1024  --feature_dim 16 --entity_dim  16  --message "1024-16"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 5  --batch_size 2048  --feature_dim 16 --entity_dim  16  --message "2048-16"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 6  --batch_size 4096  --feature_dim 16 --entity_dim  16  --message "4096-16"  --loss "pointneg" &
#python run_worldModel_kuairand.py --tau 0  --cuda 6  --batch_size 10000 --feature_dim 16 --entity_dim  16  --message "10000-16"  --loss "pointneg" &

#python run_worldModel_yahoo.py --tau 0  --cuda 0  --batch_size 64 --feature_dim 4  --entity_dim  4 --message "64-4"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 1  --batch_size 128 --feature_dim 4  --entity_dim  4 --message "128-4"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 2  --batch_size 256 --feature_dim 4  --entity_dim  4 --message "256-4"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 3  --batch_size 512 --feature_dim 4  --entity_dim  4 --message "512-4"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 4  --batch_size 1024 --feature_dim 4 --entity_dim  4  --message "1024-4"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 5  --batch_size 2048 --feature_dim 4 --entity_dim  4  --message "2048-4"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 6  --batch_size 4096 --feature_dim 4 --entity_dim  4  --message "4096-4"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 6  --batch_size 10000 --feature_dim 4 --entity_dim 4   --message "10000-4"  --loss "pointneg" &
#
#python run_worldModel_yahoo.py --tau 0  --cuda 0  --batch_size 64 --feature_dim 8  --entity_dim  8 --message "64-8"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 1  --batch_size 128 --feature_dim 8  --entity_dim 8 --message "128-8"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 2  --batch_size 256 --feature_dim 8  --entity_dim 8 --message "256-8"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 3  --batch_size 512 --feature_dim 8  --entity_dim 8 --message "512-8"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 4  --batch_size 1024 --feature_dim 8 --entity_dim 8  --message "1024-8"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 5  --batch_size 2048 --feature_dim 8 --entity_dim 8  --message "2048-8"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 6  --batch_size 4096 --feature_dim 8 --entity_dim 8  --message "4096-8"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 6  --batch_size 10000 --feature_dim 8 --entity_dim 8  --message "10000-8"  --loss "pointneg" &
#
#python run_worldModel_yahoo.py --tau 0  --cuda 0  --batch_size 64    --feature_dim 16 --entity_dim  16 --message "64-16"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 1  --batch_size 128   --feature_dim 16 --entity_dim  16  --message "128-16"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 2  --batch_size 256   --feature_dim 16 --entity_dim  16  --message "256-16"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 3  --batch_size 512   --feature_dim 16 --entity_dim  16  --message "512-16"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 4  --batch_size 1024  --feature_dim 16 --entity_dim  16  --message "1024-16"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 5  --batch_size 2048  --feature_dim 16 --entity_dim  16  --message "2048-16"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 6  --batch_size 4096  --feature_dim 16 --entity_dim  16  --message "4096-16"  --loss "pointneg" &
#python run_worldModel_yahoo.py --tau 0  --cuda 6  --batch_size 10000 --feature_dim 16 --entity_dim  16  --message "10000-16"  --loss "pointneg" &



#python run_worldModel_coat.py --tau 0  --cuda 0  --batch_size 64 --feature_dim 4  --entity_dim  4 --message "64-4"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 1  --batch_size 128 --feature_dim 4  --entity_dim  4 --message "128-4"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 2  --batch_size 256 --feature_dim 4  --entity_dim  4 --message "256-4"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 3  --batch_size 512 --feature_dim 4  --entity_dim  4 --message "512-4"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 4  --batch_size 1024 --feature_dim 4 --entity_dim  4  --message "1024-4"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 5  --batch_size 2048 --feature_dim 4 --entity_dim  4  --message "2048-4"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 6  --batch_size 4096 --feature_dim 4 --entity_dim  4  --message "4096-4"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 6  --batch_size 10000 --feature_dim 4 --entity_dim 4   --message "10000-4"  --loss "pointneg" &

#python run_worldModel_coat.py --tau 0  --cuda 0  --batch_size 64 --feature_dim 8  --entity_dim  8 --message "64-8"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 1  --batch_size 128 --feature_dim 8  --entity_dim 8 --message "128-8"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 2  --batch_size 256 --feature_dim 8  --entity_dim 8 --message "256-8"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 3  --batch_size 512 --feature_dim 8  --entity_dim 8 --message "512-8"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 4  --batch_size 1024 --feature_dim 8 --entity_dim 8  --message "1024-8"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 5  --batch_size 2048 --feature_dim 8 --entity_dim 8  --message "2048-8"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 6  --batch_size 4096 --feature_dim 8 --entity_dim 8  --message "4096-8"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 6  --batch_size 10000 --feature_dim 8 --entity_dim 8  --message "10000-8"  --loss "pointneg" &

#python run_worldModel_coat.py --tau 0  --cuda 0  --batch_size 64    --feature_dim 16 --entity_dim  16 --message "64-16"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 1  --batch_size 128   --feature_dim 16 --entity_dim  16  --message "128-16"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 2  --batch_size 256   --feature_dim 16 --entity_dim  16  --message "256-16"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 3  --batch_size 512   --feature_dim 16 --entity_dim  16  --message "512-16"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 4  --batch_size 1024  --feature_dim 16 --entity_dim  16  --message "1024-16"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 5  --batch_size 2048  --feature_dim 16 --entity_dim  16  --message "2048-16"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 6  --batch_size 4096  --feature_dim 16 --entity_dim  16  --message "4096-16"  --loss "pointneg" &
#python run_worldModel_coat.py --tau 0  --cuda 6  --batch_size 10000 --feature_dim 16 --entity_dim  16  --message "10000-16"  --loss "pointneg" &