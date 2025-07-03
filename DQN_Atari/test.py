import gymnasium as gym
import torch
import numpy as np

from model import QNetwork
from utils import preprocess
from gymnasium.wrappers import FrameStack

def evaluate(env, model, device, episodes=10):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        obs = preprocess(state)
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax(dim=1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            obs = preprocess(next_state)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards)

def main():
    env = gym.make("ALE/Breakout-v5", obs_type="grayscale", render_mode=None)
    env = FrameStack(env, num_stack=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QNetwork(state_dim=(4, 84, 84), action_dim=env.action_space.n).to(device)
    model.load_state_dict(torch.load('./checkpoints/model_state_dict_6.pth', map_location=device))
    model.eval() 

    avg_reward = evaluate(env, model, device, episodes=10)
    print(f"Average Reward over 10 episodes: {avg_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
