import torch
import numpy as np
import wandb
import yaml
import random
import os
from pathlib import Path

_USER_DATA = {'GHK': {'WANDB_ID': '6ec2335e1f6ce27570b1c7a53c2ad085a62f28fc',
                     'PATH_BASE': '/content'}
             'HZL': {'WANDB_ID': '22aca2ffa6c7ca44c7a0a98bfe68eddbcb0ff72b',
                     'PATH_BASE': '/workspace'}
                     }
_USER_CURRENT = 'HZL' #set as you before any operation
PARENT_PATH = Path(_USER_DATA[_USER_CURRENT]['PATH_BASE']) #/content
WANDB_ID = _USER_DATA[_USER_CURRENT]['WANDB_ID']
#check: numworkers, wandb.
TRAINING = True
CKPT_LOAD = False
EXP_NAME = 'WSS_SPECTRAL_LEARNING'
NUM_WORKERS = 1

#SETTINGS
if TRAINING:
    WANDB = 'TRAIN'
    wandb.login(key=WANDB_ID)
else:
    WANDB = 0
EPOCH = 500
CKPT_NAME = f'{EXP_NAME}_ms'
CKPT_TEST = PARENT_PATH / f'wss/ckpt/{CKPT_NAME}.pth'
DATASET_TYPE = 'WAVETABLE'
BLOCK_STYLE = 'CONV1D'
DECODER_STYLE = 'SPECTRAL_SEPARATED'
LOSS_SCHEDULE = True
SUB_DIM = 2
#################################################################

basic_shapes = [
    ('sin', (0, 0)),
    ('tri', (0, 3)),
    ('pul', (3, 0)),
    ('saw', (3, 3)),
    #('pwa', (-1, 0)),
    #('pwb', (-1, 1)),
    #('sap', (0, 8)),
]

serum_sub_A = [
    ('4088', (0, 0)),
    ('BottleBlow', (0, 3)),
    ('Acid', (3, 0)),
    ('Debussy', (3, 3)),
    #('pwa', (-1, 0)),
    #('pwb', (-1, 1)),
    #('sap', (0, 8)),
]

serum_sub_B = [
    ('4088', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('BottleBlow', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Acid', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Debussy', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
]

serum_sub2_B = [
    ('4088', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('BottleBlow', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Acid', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Debussy', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('AlienSpectral [SN]', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Dist Fwapper SQ', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('FFT_SQUEAL', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Evolution', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Evol Sweep', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Evol Longreece', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Dull_toy', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Dist WaTech', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Dist d00t', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Dist C2', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Dist Bass Dropper', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('Dist 8bit Fwap', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('DirtySaw', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('CrushWub', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
]

nsynth_all_B = [
    ('bass', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('brass', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('flute', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('guitar', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('keyboard', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('mallet', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('organ', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('reed', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('string', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('lead', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('vocal', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    # Add more instruments in the same tuple structure
]

nsynth_sub_B = [
    ('bass', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('organ', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
    ('string', [0]*SUB_DIM, [0]*SUB_DIM, [5]*SUB_DIM, [0]*SUB_DIM,),
]

if DATASET_TYPE=='WAVETABLE': WAVEFORMS = serum_sub2_B
elif DATASET_TYPE == 'PLAY': WAVEFORMS = nsynth_sub_B #Conditions we use
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

MU_W = switcher(WAVEFORMS, 1, 3)
LOGVAR_W = switcher(WAVEFORMS, 2, 4)

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
SR = 48000
RAW_LEN = 2**9
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

#YAML
if WANDB == "SWEEP":
    N_SIDE_LAYER = config.N_SIDE_LAYER
    B1 = config.B1
    B2 = config.B2
    LR = config.LR
else:
    N_SIDE_LAYER = config['N_SIDE_LAYER']
    B1 = config['B1']
    B2 = config['B2']
    LR = config['LR']

if not LOSS_SCHEDULE: LR = LR//10

W_VAR = np.log(0.7) #small value
ENC_H = [1, 16, 32, 64, 128, 256, 512]
ENC_K = [5, 5, 8, 8, 4, 2]
ENC_S = [4, 4, 4, 4, 2, 2]
DEC_H = [512, 256, 128, 64, 32, 16, 8]
DEC_K = [4, 8, 8, 5, 5, 2]
DEC_S = [2, 3, 3, 3, 3, 2]
LATENT_LEN = N_CONDS*SUB_DIM
RES_BLOCK_CONV_NUM = 3


if WANDB == 'TRAIN':
    wandb.init(
        # set the wandb project where this run will be logged
        project = EXP_NAME,
        
        #track hyperparameters and run metadata
        config={
        "B1": B1,
        "B2": B2,
        "LR": LR,
        "N_SIDE_LAYER": N_SIDE_LAYER,
        }
    )

if WANDB:
    CKPT_PATH = PARENT_PATH / f'wss/ckpt/{EXP_NAME}_{wandb.run.name}.pth'
else:
    while True:
        k = random.randint(0,2**12)
        if k % 5 != 0: break
    CKPT_PATH = PARENT_PATH / f'wss/ckpt/{EXP_NAME}_{k}.pth'
GPU_NUM = 0
DEVICE = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

