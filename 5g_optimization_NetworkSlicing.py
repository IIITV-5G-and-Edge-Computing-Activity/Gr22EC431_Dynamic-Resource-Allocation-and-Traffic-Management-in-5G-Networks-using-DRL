import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO

# Define the 5G Network Environment
class FiveGNetworkEnv(gym.Env):
    def __init__(self):
        super(FiveGNetworkEnv, self).__init__()
        
        # Define action space: Allocate resources (e.g., 3 slices)
        self.action_space = spaces.Discrete(3)  # 0: Slice 1, 1: Slice 2, 2: Slice 3
        
        # Define state space: Bandwidth, Latency, and QoS requirements
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # Initial state
        self.state = np.array([0.5, 0.5, 0.5])  # Example: Initial Bandwidth, Latency, QoS
        
        # Reward metrics
        self.bandwidth_capacity = 1.0
        self.latency_threshold = 0.2
        self.qos_target = 0.8

    def reset(self):
        self.state = np.array([0.5, 0.5, 0.5])  # Reset to initial state
        return self.state

    def step(self, action):
        # Simulate action effects
        bandwidth, latency, qos = self.state
        
        # Example: Modify state based on action
        if action == 0:  # Allocate more resources to Slice 1
            bandwidth = min(self.bandwidth_capacity, bandwidth + 0.1)
            latency = max(0, latency - 0.05)
            qos += 0.05
        elif action == 1:  # Allocate to Slice 2
            bandwidth = min(self.bandwidth_capacity, bandwidth + 0.05)
            latency = max(0, latency - 0.02)
            qos += 0.03
        else:  # Allocate to Slice 3
            bandwidth = max(0, bandwidth - 0.1)
            latency += 0.03
            qos += 0.02

        # Update state
        self.state = np.array([bandwidth, latency, qos])

        # Calculate reward
        reward = 0
        if latency <= self.latency_threshold and qos >= self.qos_target:
            reward = 1  # Favorable condition

        # Check if the episode is done
        done = qos >= 1.0

        return self.state, reward, done, {}

    def render(self, mode="human"):
        print(f"State: {self.state}")

# Create the environment
env = FiveGNetworkEnv()

# Train the PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test the trained agent
state = env.reset()
for step in range(10):
    action, _ = model.predict(state)
    state, reward, done, _ = env.step(action)
    env.render()
    if done:
        print("Optimization completed!")
        break
