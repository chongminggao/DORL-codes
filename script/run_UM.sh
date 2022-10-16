##lab9 ================================
#python run_worldModel_coat.py --tau 0  --cuda 2 --message "point"  --loss "point" &
#python run_worldModel_coat.py --tau 0  --cuda 3 --message "pair"  --loss "pair" &
#python run_worldModel_coat.py --tau 0  --cuda 4 --message "pp"  --loss "pp" &
#python run_worldModel_coat.py --tau 0  --cuda 5 --message "pointneg"  --loss "pointneg" &
#
#python run_worldModel_coat.py --tau 0  --cuda 2 --message "point sgd"  --loss "point" --optimizer sgd &
#python run_worldModel_coat.py --tau 0  --cuda 3 --message "pair sgd"  --loss "pair" --optimizer sgd &
#python run_worldModel_coat.py --tau 0  --cuda 4 --message "pp sgd"  --loss "pp" --optimizer sgd &
#python run_worldModel_coat.py --tau 0  --cuda 5 --message "pointneg sgd"  --loss "pointneg" --optimizer sgd &
#
#python run_worldModel_yahoo.py --tau 0 --cuda 2  --message "point"  --loss "point" &
#python run_worldModel_yahoo.py --tau 0 --cuda 3  --message "pair"  --loss "pair" &
#python run_worldModel_yahoo.py --tau 0 --cuda 4  --message "pp"  --loss "pp" &
#python run_worldModel_yahoo.py --tau 0 --cuda 5  --message "pointneg"  --loss "pointneg" &
#
#python run_worldModel_yahoo.py --tau 0 --cuda 2  --message "point sgd"  --loss "point" --optimizer sgd &
#python run_worldModel_yahoo.py --tau 0 --cuda 3  --message "pair sgd"  --loss "pair" --optimizer sgd &
#python run_worldModel_yahoo.py --tau 0 --cuda 4  --message "pp sgd"  --loss "pp" --optimizer sgd &
#python run_worldModel_yahoo.py --tau 0 --cuda 5  --message "pointneg sgd"  --loss "pointneg" --optimizer sgd &


#lab9 ================================
# todo: neg_in_train = False.
python run_worldModel_kuairand.py --tau 0  --cuda 2 --message "point_oo"  --loss "point" &
python run_worldModel_kuairand.py --tau 0  --cuda 3 --message "pair_oo"  --loss "pair" &
python run_worldModel_kuairand.py --tau 0  --cuda 4 --message "pp_oo"  --loss "pp" &
python run_worldModel_kuairand.py --tau 0  --cuda 5 --message "pointneg_oo"  --loss "pointneg" &

#python run_worldModel_kuairand.py --tau 0  --cuda 2 --message "point sgd"  --loss "point" --optimizer sgd &
#python run_worldModel_kuairand.py --tau 0  --cuda 3 --message "pair sgd"  --loss "pair" --optimizer sgd &
#python run_worldModel_kuairand.py --tau 0  --cuda 4 --message "pp sgd"  --loss "pp" --optimizer sgd &
#python run_worldModel_kuairand.py --tau 0  --cuda 5 --message "pointneg sgd"  --loss "pointneg" --optimizer sgd &

#python run_worldModel_kuairec.py --tau 0 --cuda 2  --message "point"  --loss "point" &
#python run_worldModel_kuairec.py --tau 0 --cuda 3  --message "pair"  --loss "pair" &
#python run_worldModel_kuairec.py --tau 0 --cuda 4  --message "pp"  --loss "pp" &
#python run_worldModel_kuairec.py --tau 0 --cuda 5  --message "pointneg"  --loss "pointneg" &

#python run_worldModel_kuairec.py --tau 0 --cuda 2  --message "point sgd"  --loss "point" --optimizer sgd &
#python run_worldModel_kuairec.py --tau 0 --cuda 3  --message "pair sgd"  --loss "pair" --optimizer sgd &
#python run_worldModel_kuairec.py --tau 0 --cuda 4  --message "pp sgd"  --loss "pp" --optimizer sgd &
#python run_worldModel_kuairec.py --tau 0 --cuda 5  --message "pointneg sgd"  --loss "pointneg" --optimizer sgd &