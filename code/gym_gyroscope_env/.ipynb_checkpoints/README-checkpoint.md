# Double gimbal control moment gyroscope Gym environment

- To create new environments for Gym

https://github.com/openai/gym/blob/master/docs/creating-environments.md

- To import a specific environment: 

from gym.envs.registration import register
register(
    id = 'GyroscopeEnv-v0', 
    entry_point = 'gym_gyroscope_env.gyroscope_env:GyroscopeEnv'
)

- To use using the env_fn from the custom_functions module function that allows to pass arguments to the class:

reward_args = {'qx1':6,'qx2':0.01,'qx3':6,'qx4':0.01,'pu1':0.05,'pu2':0.05}
env_fn = partial(env_fn,env_name = 'GyroscopeEnv-v0',reward_type = 'Quadratic', reward_args = reward_args)

- To install: 

pip3 install -e .

