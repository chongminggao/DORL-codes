
This Readme is not done yet. It is only for an illustration purpose.

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

4. Install the required modules from pip.

    ```bash
    sh install.sh
    ```
   Install the tianshou package from my forked version:
   ```bash
   cd src
   git clone https://github.com/chongminggao/tianshou.git
   cd ..
   ```


## Download the data
1. Download the compressed dataset

    ```bash 
    wget https://chongming.myds.me:61364/DORL/environments.tar.gz
    ```
   or you can manually download it from this website:
   https://rec.ustc.edu.cn/share/9fe264f0-ae09-11ed-b9ef-ed1045d76757
   


2. Uncompress the downloaded `environments.tar.gz` and put the files to their corresponding positions.

   ```bash
   tar -zxvf environments.tar.gz
   ```

If things go well, you can run the following examples now.

---
## Examples to run the code

The argument `env` of all experiments can be set to one of the four environments: `CoatEnv-v0, Yahoo-v0, KuaiEnv-v0, KuaiRand-v0`. The former two datasets (coat and yahoo) are small so the models can run very quickly.


#### 1. Run traditional DQN-based baselines (e.g., BCQ or SQN) 

The examples are illustrated on  `CoatEnv-v0` , which runs quickly.

**Step 1:** You should run the static user model (DeepFM) to get the embedding of users and items. 

```shell
  python run_worldModel_ensemble.py --env CoatEnv-v0  --cuda 0 --epoch 5 --tau 0 --loss "pointneg" --message "pointneg"
```
Where argument `message` is a maker for naming the saved files. For example, by setting `--message "pointneg"`, you will get all saved files with names contains `"pointneg"`. Here, "pointneg" is just a loss that we use, and you can replace this message to any other words that you like.   

**Step 2:** Run the RL policy. By default, we use the user/item embeddings trained from the last step. There are many policies. Here, we run the BCQ and SQN for example:

Run BCQ:
   ```bash 
   python run_Policy_BCQ.py  --env CoatEnv-v0  --seed 0 --cuda 0    --which_tracker avg --reward_handle "cat"  --unlikely-action-threshold 0.6 --window_size 3 --read_message "pointneg"  --message "BCQ"
   ```
Run SQN:
   ```bash
   python run_Policy_SQN.py  --env CoatEnv-v0   --seed 0 --cuda 0   --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg"  --message "SQN"
   ```



#### 2. Run model-based method, i.e., DORL, MBPO, and MOPO.

The examples are illustrated on  `KuaiRec-v0` , which runs a little bit slowly compared to codes above.

DORL can only be implemented it on the `KuaiEnv-v0` or `KuaiRand-v0` environments, since the two datasets contains recommendation logs that Coat and Yahoo datasets do not have. 

The procedure still contains two steps: 

**Step 1:** run the static user model (DeepFM) to get the embedding of users and items.

```shell
python run_worldModel_ensemble.py --env KuaiEnv-v0  --cuda 0 --epoch 5 --loss "pointneg" --message "pointneg"
```

Where argument `message` is a maker for naming the saved files. For example, by setting `--message "pointneg"`, you will get all saved files with names contains `"pointneg"`. Here, "pointneg" is just a loss that we use, and you can replace this message to any other words that you like.   

**Step 2:** Run the RL policy. Use the argument `--read_message "pointneg"` to make sure that the embeddings and rewards are generated from the model trained in step 1.

Run DORL (Our proposed method):

   ```bash 
python run_Policy_Main.py --env KuaiEnv-v0  --seed 0 --cuda 0  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_variance 0.01 --lambda_entropy 0.05  --window_size 3 --read_message "pointneg"  --message "DORL" &
   ```

Run MBPO (the vanilla model-based offline RL method):

```shell
python run_Policy_Main.py --env KuaiEnv-v0  --seed 0 --cuda 0   --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_variance 0   --lambda_entropy 0    --window_size 3 --read_message "pointneg"  --message "MBPO" &
```

Run MOPO (the model-based RL method considering distributional shift):

```shell
python run_Policy_Main.py --env KuaiEnv-v0  --seed 0 --cuda 0   --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0    --window_size 3 --read_message "pointneg"  --message "MOPO" &
```





