
## Installation

1. Clone this git repository and change directory to this repository:

    ```shell
    git clone https://github.com/chongminggao/Rethink_RL4RS.git
    cd Rethink_RL4RS/
    ```

2. A new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) is suggested. 

    ```bash
    conda create --name myRL python=3.10 -y
    ```

3. Activate the newly created environment.

    ```bash
    conda activate myRL
    ```

4. Install the required modules

    ```bash
    sh install.sh
    ```
   Install the tianshou package:
   ```bash
   cd src
   git clone https://github.com/thu-ml/tianshou.git
   cd tianshou
   git checkout 0f59e38
   cd ../..
   ```


## Download the data
1. Download the compressed dataset

    ```bash 
    wget https://chongming.myds.me:61364/DORL/environments.tar.gz
    ```



2. Uncompress the downloaded `environments.tar.gz` and put the files to their corresponding positions.

   ```bash
   tar -zxvf environments.tar.gz
   ```

If things go well, you can run the following examples now.

---
## Examples to run the code

The argument `env` of all experiments can be set to one of the four environments: `CoatEnv-v0, Yahoo-v0, KuaiEnv-v0, KuaiRand-v0`. The former two datasets (coat and yahoo) are small so the models can run very quickly.
So we use `CoatEnv-v0` as an example.

Step 1: You should run the static user model (DeepFM) to get the embedding of users and items. 
```shell
  python run_worldModel_ensemble.py --env CoatEnv-v0  --cuda 0 --epoch 5 --tau 0 --loss "pointneg" --message "pointneg"
```
Where argument `message` is a maker for naming the saved files. For example, by setting `--message "pointneg"`, you will get all saved files with names contains `"pointneg"`.   

Run DORL：
   ```bash 
    python run_Policy_SQN.py  --env CoatEnv-v0   --seed 0 --cuda 0   --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg"  --message "SQN" &
    python run_Policy_BCQ.py  --env CoatEnv-v0  --seed 0 --cuda 4    --which_tracker avg --reward_handle "cat"  --unlikely-action-threshold 0.6 --window_size 3 --read_message "pointneg"  --message "BCQ" &
   ```
 





