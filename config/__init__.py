import torch
import numpy as np
import wandb
import yaml
import random
import os
from pathlib import Path

_USER_CURRENT = 'HZL' #set as you before any operation
PARENT_PATH = Path('/workspace') #/content
WANDB_ID = '22aca2ffa6c7ca44c7a0a98bfe68eddbcb0ff72b'
torch.autograd.set_detect_anomaly(True)
#check: numworkers, wandb.
TRAINING = 'TRAIN'
CKPT_LOAD = False
if CKPT_LOAD: STARTING_EPOCH = 1500
NUM_WORKERS = 24
#Major Model Settings

#SETTINGS
if TRAINING == 'SWEEP':
    WANDB = 'SWEEP'
    wandb.run.name = f'{TINY}{LEARN_PRIORS}{SET}'
elif TRAINING:
    WANDB = 'TRAIN'
    wandb.login(key=WANDB_ID)
else:
    WANDB = 0

AB_D, AB_L, AB_S = 1, 0, 0

EPOCH = 5000
TEST_NAME = '1'
TINY = 0
LEARN_PRIORS = 0
if LEARN_PRIORS:
    PRIOR_COEF = 17
EXP_NAME = f'WSS_ISMIR_AB_{AB_D}{AB_L}{AB_S}'
CKPT_NAME = 'WSS_ISMIR_SE_S0_PL0_SET0'
CKPT_TEST = PARENT_PATH / f'wss/ckpt/{CKPT_NAME}.pth'
DATASET_TYPE = 'WAVETABLE'
BLOCK_STYLE = 'CONV1D'
DECODER_STYLE = 'SPECTRAL_SEPARATED'
LOSS_SCHEDULE = True #0.001 -> 0.0001
SUB_DIM = 2
#################################################################

basic_shapes = [
    ('sin', (0, 0)),
    ('tri', (0, 3)),
    ('pul', (3, 0)),
    ('saw', (3, 3)),
]

serum_sub_A = [
    ('4088', (0, 0)),
    ('BottleBlow', (0, 3)),
    ('Acid', (3, 0)),
    ('Debussy', (3, 3)),
]

serum_sub_B = []
serum_sub2_B = [
    ('4088', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('BottleBlow', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('Acid', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('Debussy', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('AlienSpectral [SN]', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('Dist Fwapper SQ', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('FFT_SQUEAL', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('Evolution', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('Evol Sweep', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('Evol Longreece', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('Dull_toy', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('Dist WaTech', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('Dist d00t', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('Dist C2', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('Dist Bass Dropper', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('Dist 8bit Fwap', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('DirtySaw', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('CrushWub', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
]

internal_ = [
    ('PNO', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('STR', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('WND', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('GTR', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
]

waveedit = [
    ('softwaves', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
    ('rect', [0.]*SUB_DIM, [0.]*SUB_DIM, [5.]*SUB_DIM, [0.]*SUB_DIM,),
]

###########
if DATASET_TYPE == 'WAVETABLE': WAVEFORMS = serum_sub2_B #internal_
elif DATASET_TYPE == 'PLAY': WAVEFORMS = nsynth_all_B #Conditions we use
N_CONDS = len(WAVEFORMS)

WAVEFORM_NAMES = [i[0] for i in WAVEFORMS]

def switcher(conds, false, true):
    out = []
    for i in range(N_CONDS):
        sub_out = []
        for j in range(N_CONDS):
            if i==j:
                sub_out = sub_out + conds[j][true]
            else:
                sub_out = sub_out + conds[j][false]
        out.append(sub_out)
    return out

MU_Z = switcher(WAVEFORMS, 1, 3)
LOGVAR_Z = switcher(WAVEFORMS, 2, 4)

nsynth_method_dictionary = {'bass': 0,
                            'synthetic': 1,
                            'acoustic': 2,
                            'electronic': 3,
                            }

#################################################################
if WANDB == 'SWEEP':
    wandb.init(
        # set the wandb project where this run will be logged
        project = EXP_NAME,
    )
    config = wandb.config
else:
    with open(PARENT_PATH / f'wss/config/config.yaml', 'r') as stream:
        config = yaml.safe_load(stream) 
##DATA
SR = 16000
RAW_LEN = 2**10
X_DIM = 2**9
Y_DIM = N_CONDS
POS_DIM = 1
N_FFT = RAW_LEN//2 + 1

##MODEL, LOSS & LEARNING
BS = 64 #batch size.
W_DIM = N_CONDS*SUB_DIM
LEAKY_RELU_P = 0.2
DROPOUT_RATE = 0.2
EPSILON = 1e-8
LOGVAR = 0
NORMALISED_ENERGY = 1
BETA = 1/256
serum_sub_B = 0
#YAML
if WANDB == "SWEEP":
    SPECTRAL_LOSS_COEF = config.SPECTRAL_LOSS_COEF
    WAVEFORM_LOSS_COEF = config.WAVEFORM_LOSS_COEF
    SEMANTIC_LOSS_COEF = config.SEMANTIC_LOSS_COEF
    PHASE_LOSS_COEF = config.PHASE_LOSS_COEF
    NOISE_LOSS_COEF = config.NOISE_LOSS_COEF
    KL_LOSS_COEF = config.KL_LOSS_COEF
    LR = config.LR
    SEED = config.SEED
    WAVEFORM_LOSS_MULTIPLIER = config.WAVEFORM_LOSS_MULTIPLIER
    WAVEFORM_LOSS_DECREASE_RATE = config.WAVEFORM_LOSS_DECREASE_RATE
else:
    SPECTRAL_LOSS_COEF = config['SPECTRAL_LOSS_COEF']
    WAVEFORM_LOSS_COEF = config['WAVEFORM_LOSS_COEF']
    SEMANTIC_LOSS_COEF = config['SEMANTIC_LOSS_COEF']
    PHASE_LOSS_COEF = config['PHASE_LOSS_COEF']
    NOISE_LOSS_COEF = config['NOISE_LOSS_COEF']
    KL_LOSS_COEF = config['KL_LOSS_COEF']
    LR = config['LR']
    SEED = config['SEED']
    WAVEFORM_LOSS_MULTIPLIER = config['WAVEFORM_LOSS_MULTIPLIER']
    WAVEFORM_LOSS_DECREASE_RATE = config['WAVEFORM_LOSS_DECREASE_RATE']

if AB_L == 0:
    PHASE_LOSS_COEF = 0
    NOISE_LOSS_COEF = 0
    SEMANTIC_LOSS_COEF = 0

if not LOSS_SCHEDULE: LR = LR//10

W_VAR = np.log(0.7) #small value
if TINY:
    ENC_H = [1, 4, 8, 16, 32, 64, 128]
    ENC_K = [3, 3, 5, 5, 2, 2]
    ENC_S = [4, 4, 4, 4, 2, 2]
    DEC_H = [128, 64, 32, 16, 8, 4, 2]
    DEC_K = [4, 8, 8, 5, 5, 2]
    DEC_S = [2, 3, 3, 3, 3, 2]
else:
    ENC_H = [1, 16, 32, 64, 128, 256, 512]
    ENC_K = [5, 5, 8, 8, 4, 2]
    ENC_S = [4, 4, 4, 4, 2, 2]
    DEC_H = [512, 256, 128, 64, 32, 16, 8]
    DEC_K = [4, 8, 8, 5, 5, 2]
    DEC_S = [2, 3, 3, 3, 3, 2]
LATENT_LEN = N_CONDS*SUB_DIM #固定
if AB_D: SEMANTIC_CONDITION_LEN = 5
else: SEMANTIC_CONDITION_LEN = 0
RES_BLOCK_CONV_NUM = 3

GPU_NUM = 12
DEVICE = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    # Set the random seed for PyTorch on CPU and GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for Python's built-in random module
    random.seed(seed)

# Example usage:
random_seed = SEED  # Choose any integer value as the random seed
set_seed(random_seed)

torch.set_num_threads(1)