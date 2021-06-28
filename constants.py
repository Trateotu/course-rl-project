from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

POS_VAL = 20
GOAL_VAL = 10
OBSTACLE_VAL = 1  # FIXED

# GRID_SIZE = 10  # 20

HORIZON_MULTIPLIER = 3

RWD_DEATH = 0

EPISODES = 10 # 20000  # 50000
EPISODES_TEST = 1000

EPOCHS = 1000000 # 100

# PPO constants
EPISODES_PER_UPDATE = 100
MODEL_SAVE_FREQ = round(EPISODES / 10)

# MAZES_BATCH_SIZE = 5

# LOG_DIR = './logs/exp6'