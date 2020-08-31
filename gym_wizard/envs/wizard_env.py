import gym
from gym import error, spaces, utils
from gym.utils import seeding

class WizardEnv(gym.Env):
    metadata = {'render.modes':['array']}

    def __init__(self):
        pass
    
    def reset(self):
        print("Reset wizard")

    def step(self, action):
        pass

    def render(self, mode='array', close=False):
        pass
