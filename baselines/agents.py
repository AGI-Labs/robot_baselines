import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):
    def reset(self):
        """ Resets agent at start of trajectory """

    def forward(self, image, robot_state):
        raise NotImplementedError

class ClosedLoopAgent(Agent):
    def __init__(self, policy_network):
        super().__init__()
        self._pi = policy_network


    def forward(self, image, robot_state):
        return self._pi(image, robot_state)
