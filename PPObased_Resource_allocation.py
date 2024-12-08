import numpy as np
from gym import spaces, Env
from stable_baselines3 import PPO

class ResourceAllocationEnv(Env):
    def _init_(self):
        super(ResourceAllocationEnv, self)._init_()
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.state = np.random.rand(3)
        self.total_resources = 1.0

    def reset(self):
        self.state = np.random.rand(3)
        return self.state

    def step(self, action):
        allocation = np.clip(action / np.sum(action), 0, 1) * self.total_resources
        rewards = -np.abs(self.state - allocation)
        self.state = np.random.rand(3)
        return self.state, np.sum(rewards), False, {}

env = ResourceAllocationEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

state = env.reset()
for _ in range(5):
    action, _ = model.predict(state)
    state, reward, _, _ = env.step(action)
    print(f"State: {state}, Action: {action}, Reward: {reward}")