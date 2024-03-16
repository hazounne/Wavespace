from .dataset import *
from .model import *

#Dataset configuration.
if (WAVEFORMS == serum_sub2_B) or (WAVEFORMS == serum_sub_B): DATA_FOLDERS = [PARENT_PATH / 'wss/SerumDataset']
elif WAVEFORMS == internal_ : DATA_FOLDERS = [PARENT_PATH / 'wss/internal_wavetable']
elif WAVEFORMS == waveedit: DATA_FOLDERS = [PARENT_PATH / 'wss/WaveEditDataset/processed_upscaled']
DATASETS = collapse(DATA_FOLDERS)
