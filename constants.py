from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

POS_VAL = 20
GOAL_VAL = 10
OBSTACLE_VAL = 1  # FIXED

GRID_SIZE = 22  # FIXED

RWD_DEATH = 0

EPISODES = 50000

LOG_DIR = './logs/exp5'