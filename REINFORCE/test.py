import gymnasium as gym
import torch
import numpy as np

from model import PolicyNetwork
from torch.distributions import Categorical



def evaluate(env, model, device, episodes=10):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.from_numpy(state).to(device).float()
            with torch.no_grad():
                probs = model(state_tensor)
                dist = Categorical(probs=probs)
                action = dist.sample().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards)

def main():
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolicyNetwork(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n).to(device)
    model.load_state_dict(torch.load('./checkpoints/model_state_dict_1.pth', map_location=device))
    model.eval() 

    avg_reward = evaluate(env, model, device, episodes=10)
    print(f"Average Reward over 10 episodes: {avg_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
