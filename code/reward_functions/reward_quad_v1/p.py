import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy.integrate import solve_ivp

import gym_gyroscope_env
import spinup
import torch
from functools import partial

from custom_functions.custom_functions import env_fn 
from custom_functions.custom_functions import create_env
from custom_functions.custom_functions import load_agent
from custom_functions.custom_functions import test_agent
from custom_functions.custom_functions import plot_test
from custom_functions.custom_functions import evaluate_control


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 200,
    'seed': 2
}
reward_func = 'Power'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0,
    'p': 0.05
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
spinup.ddpg_pytorch(env_fn_, 
                    ac_kwargs = dict(hidden_sizes=[128,32], activation=torch.nn.ReLU), 
                    seed = 0, 
                    steps_per_epoch = 1500, 
                    epochs = 5000, 
                    replay_size = 1000000, 
                    gamma = 0.995, 
                    polyak = 0.995, 
                    pi_lr = 0.0025,
                    q_lr = 0.0025,
                    batch_size = 100, 
                    start_steps = 10000,
                    act_noise = 0.1,
                    max_ep_len = 200, 
                    logger_kwargs = dict(output_dir='ddpg_p005_len200', exp_name='ddpg_p005_len200'))
                    
# Env
env_name = 'GyroscopeEnv-v1'
env = create_env(env_name)

# num and set
num_test = 10000
states = np.genfromtxt('states10k.csv', delimiter=',')

# Init dataframe
agent_paths = ['ddpg_p005_len200']
t_end = 10
ss_bound = 0.25

# Loop dataframe
for idx, agent_path in enumerate(agent_paths):
    print(agent_path)
    agent = load_agent(agent_path)
    if idx == 0:
        metrics = evaluate_control(env,agent,agent_path,t_end,ss_bound,num_test,states)
    else:
        new_metrics = evaluate_control(env,agent,agent_path,t_end,ss_bound,num_test,states)
        metrics = metrics.append(new_metrics)
        
metrics_man = metrics.transpose()
metrics_man = metrics_man.round(4)

metrics_man.to_csv('Metrics_Plen.csv',index=True)