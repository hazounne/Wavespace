#Import prerequisites.
import os
import random
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio.transforms import Resample
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import random

SEED = 1000
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False