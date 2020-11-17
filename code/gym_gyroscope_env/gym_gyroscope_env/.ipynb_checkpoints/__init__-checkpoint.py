from gym.envs.registration import register

register(
    id = 'GyroscopeEnv-v0', 
    entry_point = 'gym_gyroscope_env.envs:GyroscopeEnv'
)
register(
    id = 'GyroscopeDiscontinuousEnv-v0', 
    entry_point = 'gym_gyroscope_env.envs:GyroscopeDiscontinuousEnv'
)
register(
    id = 'GyroscopeIntegralEnv-v0', 
    entry_point = 'gym_gyroscope_env.envs:GyroscopeIntegralEnv'
)
register(
    id = 'GyroscopeRobustEnv-v0', 
    entry_point = 'gym_gyroscope_env.envs:GyroscopeRobustEnv'
)
