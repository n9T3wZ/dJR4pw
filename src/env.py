import numpy as np
import torch
from torch.utils.data import DataLoader
from src.IP import *
from src.LTNN import *
from src.data_loader import *

class IPenv:
    def __init__(self, file_paths, num_cons, num_vars, num_cuts=1, device = torch.device("cpu"), problem_type="knapsack"):
        self.dataset = KnapsackDataset(file_paths)
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.device = device
        self.num_cons = num_cons
        self.num_vars = num_vars
        self.num_cuts = num_cuts
        self.problem_type = problem_type
        self.dataloader = iter(dataloader)

        self.score_list = []

        self.state = None
        self.reset()

    
    def reset(self):
        # Fetch a new state from the dataloader.
        try:
            self.state, self.treesize = next(self.dataloader)
            self.treesize = self.treesize.item()
        except StopIteration:
            # If dataloader is exhausted, restart it.
            self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
            self.dataloader = iter(self.dataloader)
            self.state, self.treesize = next(self.dataloader)
            self.treesize = self.treesize.item()

        return self.state
    
    def get_state(self):
        return self.state
    
    def step(self, action):
        u = action.cpu().detach().numpy().reshape(-1,)
        x = self.state.cpu().detach().numpy().reshape(-1,)
        tree_size_before_cut = self.treesize

        ip_new = vector_to_ip(x, self.num_cons, self.num_vars, self.problem_type)
        alpha, beta = ip_new.add_chvatal_cut(u)
        c = np.array(ip_new.c)
        ip_new.optimize()
        tree_size_after_cut = ip_new.treesize
        
        reward = (tree_size_before_cut - tree_size_after_cut) / tree_size_before_cut

        reward = torch.tensor(reward).to(self.device).float()
        
        # Get the next state from the dataloader.
        try:
            next_state = next(self.dataloader)[0]
        except StopIteration:
            self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
            self.dataloader = iter(self.dataloader)
            next_state =  next(self.dataloader)[0]
            done = True
        else:
            done = False
        self.state = next_state
        
        return next_state, reward, done


