


python run_worldModel_coat.py --tau 0   --message "point"  --loss "point" &
python run_worldModel_coat.py --tau 0   --message "pair"  --loss "pair" &
python run_worldModel_coat.py --tau 0   --message "pp"  --loss "pp" &


python run_worldModel_coat.py --tau 0   --message "point sgd"  --loss "point" --optimizer sgd &
python run_worldModel_coat.py --tau 0   --message "pair sgd"  --loss "pair" --optimizer sgd &
python run_worldModel_coat.py --tau 0   --message "pp sgd"  --loss "pp" --optimizer sgd &