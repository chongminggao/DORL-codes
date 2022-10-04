#lab9 ================================
python run_worldModel_coat.py --tau 0  --cuda 2 --message "point"  --loss "point" &
python run_worldModel_coat.py --tau 0  --cuda 3 --message "pair"  --loss "pair" &
python run_worldModel_coat.py --tau 0  --cuda 4 --message "pp"  --loss "pp" &
python run_worldModel_coat.py --tau 0  --cuda 5 --message "pointneg"  --loss "pointneg" &

python run_worldModel_coat.py --tau 0  --cuda 2 --message "point sgd"  --loss "point" --optimizer sgd &
python run_worldModel_coat.py --tau 0  --cuda 3 --message "pair sgd"  --loss "pair" --optimizer sgd &
python run_worldModel_coat.py --tau 0  --cuda 4 --message "pp sgd"  --loss "pp" --optimizer sgd &
python run_worldModel_coat.py --tau 0  --cuda 5 --message "pointneg sgd"  --loss "pointneg" --optimizer sgd &

python run_worldModel_yahoo.py --tau 0 --cuda 2  --message "point"  --loss "point" &
python run_worldModel_yahoo.py --tau 0 --cuda 3  --message "pair"  --loss "pair" &
python run_worldModel_yahoo.py --tau 0 --cuda 4  --message "pp"  --loss "pp" &
python run_worldModel_yahoo.py --tau 0 --cuda 5  --message "pointneg"  --loss "pointneg" &

python run_worldModel_yahoo.py --tau 0 --cuda 2  --message "point sgd"  --loss "point" --optimizer sgd &
python run_worldModel_yahoo.py --tau 0 --cuda 3  --message "pair sgd"  --loss "pair" --optimizer sgd &
python run_worldModel_yahoo.py --tau 0 --cuda 4  --message "pp sgd"  --loss "pp" --optimizer sgd &
python run_worldModel_yahoo.py --tau 0 --cuda 5  --message "pointneg sgd"  --loss "pointneg" --optimizer sgd &