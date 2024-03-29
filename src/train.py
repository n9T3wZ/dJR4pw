import numpy as np
import collections
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    @property
    def size(self):
        return len(self.buffer)

def moving_average(data, window_size=10):
    return [np.mean(data[max(0, i - window_size + 1):(i + 1)]) for i in range(len(data))]

def train(env, agent, num_samples, replay_buffer, minimal_size, batch_size, epochs=1):
    returns = []
    writer = SummaryWriter()
    global_step = 0

    for epoch in range(epochs):
        pbar = tqdm(range(num_samples), desc=f"Epoch {epoch + 1}")
        episode_return = 0
        
        for sample in pbar:
            state = env.reset()
            done = False
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            episode_return += reward.item()
            state = next_state

            global_step += 1

            if replay_buffer.size > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                b_s = torch.stack([i.reshape(-1) for i in b_s])
                b_a = torch.stack([i.reshape(-1) for i in b_a])
                b_r = torch.stack(b_r)
                b_ns = torch.stack([i.reshape(-1) for i in b_ns])
                b_d = torch.tensor(b_d, dtype=torch.int).view(-1, 1)

                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                loss = agent.update(transition_dict)

                writer.add_scalar(f'Loss/Epoch_{epoch + 1}_critic_loss', loss, global_step)

            if sample % 100 == 0 and sample > 0:
                returns.append(episode_return)
                moving_avg = np.mean(returns[-min(10, len(returns)):])

                writer.add_scalar(f'Performance/Epoch_{epoch + 1}_return', episode_return, global_step)
                writer.add_scalar(f'Performance/Epoch_{epoch + 1}_moving_average_return', moving_avg, global_step)
                
                pbar.set_postfix({"Return": episode_return, "Moving Avg Return": moving_avg})

                episode_return = 0

    writer.close()
    return returns
