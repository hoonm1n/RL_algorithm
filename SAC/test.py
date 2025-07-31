import gym
import torch
import numpy as np
from model import PolicyNetwork  

def evaluate_policy(env_name, model_path, device="cpu", episodes=10, render=False):
    env = gym.make(env_name)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    policy = PolicyNetwork(ob_dim, ac_dim).to(device)
    policy.load_state_dict(torch.load("./checkpoints/model_state_dict_halfcheetah_2.pth", map_location=device))
    policy.eval()

    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = policy(obs_tensor)
                action = torch.tanh(mu).cpu().numpy()[0]

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if render:
                env.render()

        rewards.append(total_reward)
        print(f"Episode {ep + 1}, Reward: {total_reward:.2f}")

    env.close()
    print(f"\nAverage Reward over {episodes} episodes: {np.mean(rewards):.2f}")

if __name__ == "__main__":
    evaluate_policy(
        env_name="HalfCheetah-v4",
        model_path="./checkpoints/model_state_dict_halfcheetah_2.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
        episodes=10,
        render=False 
    )
