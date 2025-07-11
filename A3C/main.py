import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from agent import A3C
from model import ActorCritic
import torch.optim as optim


def worker(global_network, optimizer, global_step, lock, worker_idx):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset(seed=1234 + worker_idx)
    worker_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = A3C(worker_env, worker_device, global_network, optimizer, global_step, lock, worker_idx, gamma=0.99,)
    agent.update(total_update_steps=550000)

    print("worker done")






def main():
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n

    global_network = ActorCritic(ob_dim, ac_dim).to(device)
    global_network.share_memory()
    
    optimizer = optim.RMSprop(global_network.parameters(), lr=7e-4, alpha=0.99, eps=1e-5, momentum=0, centered=False)

    global_step = mp.Value('i', 0)
    lock = mp.Lock()

    num_workers = 16

    processes = []
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(global_network, optimizer, global_step, lock, i))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()


    torch.save(global_network.state_dict(), './checkpoints/model_state_dict_2.pth')
    env.close()






if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()