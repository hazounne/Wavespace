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
if (WAVEFORMS == serum_sub2_B) or (WAVEFORMS == serum_sub_B): DATA_FOLDERS = [PARENT_PATH / 'wss/SerumDataset']
elif WAVEFORMS == _internal: DATA_FOLDERS = [PARENT_PATH / 'wss/internal_wavetable']
DATASETS = collapse(DATA_FOLDERS)
