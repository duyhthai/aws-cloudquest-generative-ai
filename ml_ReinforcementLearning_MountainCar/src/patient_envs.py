from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.wrappers.time_limit import TimeLimit

def PatientContinuousMountainCar():
    env = Continuous_MountainCarEnv()
    return TimeLimit(env, max_episode_steps=10000)
