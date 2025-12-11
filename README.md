
# BattleshipRL

3 reinforcement learning models (DQN, PPO, REINFORCE) trained to play Battleship

## Setup

First clone this repository with 

`git clone https://github.com/shayanhalder/battleship-reinforcement-learning.git`

Now create a Python3.10 virtual environment. 

**NOTE: Python3.10 is REQUIRED otherwise you will run into dependency issues.** 

You must already have Python3.10 installed on your system.   

MacOS download: [https://www.python.org/downloads/macos/](https://www.python.org/downloads/macos/)   
Windows download: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)

Run the following in the terminal (MacOS/Linux Bash terminal):

```
python3.10 -m venv venv
source venv/bin/activate
```

Or the following if you have Windows (Powershell terminal): 
```
python3.10 -m venv venv
venv\Scripts\Activate.ps1
```

**NOTE: `python3.10` command in the terminal should point to the Python3.10 interpreter installed on your system. It may be different if you have custom aliases installed or have modified your PATH environment variable.**

Now install the dependencies with

`pip install -r requirements.txt`

## Model Evaluation Pipeline

`project.ipynb` contains the evaluation pipeline for each RL model (DQN, PPO, REINFORCE). 

Open `project.ipynb` with VS Code (or your preferred Jupyter Notebook editor) and select `venv` as the Python kernel in the top right (should be Python3.10).

The code cells for each of the models are separated by section and automatically collapsed for readbility.

Run the cells sequentially and the evaluation graphs (number of moves to win, hit-to-miss ratio, action distribution) will be displayed in the notebook and also saved to the specified directory. 

For DQN, we trained 10 models with different hyperparameters and the best model was `dqn_models/dqn_model_9/dqn-9` so that one will be used. The evaluation graphs will be saved to `dqn_models/final_model`. The video demo GIF will be saved to `dqn_models/video-demos/dqn_final_model_video.gif`

For PPO, they will be saved to `ppo_models`. The final model, evalutation visuals, and the recording are all saved in this folder.

For REINFORCE, they will be saved to ``

**NOTE: The DQN model file size was too large to store on GitHub, so the `project.ipynb` notebook will automatically download it from Git LFS (Large File Storage) before running the evaluation code.** 

## Model Training Pipeline

### DQN 
`dqn_pipeline.ipynb` contains the training pipeline for all 10 of the DQN models. By default, all the hyperparameter sets are commented out except the one for the final DQN model we used for evaluation (dqn-9). Run all the cells sequentially to train the model from scratch and evaluate it. 

**NOTE: training all 10 DQN models from scratch will take ~4-5 hours**

### PPO
`ppo.ipynb` contains the training pipeline for the final PPO model. To train and evaluate the model from scratch, run the cells in sequential order. 

