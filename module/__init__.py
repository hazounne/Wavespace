from .dataset import *
from .model import *

#Dataset configuration.
DATA_FOLDERS = [#'/workspace/wss/SerumDataset', #SerumDataset
                #'/workspace/wss/SerumWaves'
                '/data3/NSynth/nsynth-train', #Nsynth Train Set
                #'/data3/NSynth/nsynth-vaild', #Nsynth Valid Set
                #'/data3/NSynth/nsynth-test', #Nsynth Test Set
                ]

if not WANDB: DATA_FOLDERS = ['/data3/NSynth/nsynth-test']
if DATASET_TYPE == 'WAVETABLE': DATA_FOLDERS = [PARENT_PATH / 'wss/SerumDataset']
DATASETS = collapse(DATA_FOLDERS)
