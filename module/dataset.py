from funcs import *
from config import *
import os
import torch
import torchaudio
import re
from torch.utils.data import Dataset, DataLoader, random_split
from torch.fft import rfft

'''
[order of data preparation]
data_folders
-> collapse (DATASETS from module.__init__)
-> DatasetBuilder
-> data_loader(assigned in experiment): concat each datum in the list
'''

# 데이터 로더 생성
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the list consisting of each data directory.
def collapse(data_folders:list) -> list:
  # saves all the directories
  output_file_lists = list()
  for data_folder in data_folders: #put in every wave files, put away silences
    dir_list = []
    for r, _, filenames in os.walk(data_folder):
      for filename in filenames:
        data_dir = f'{r}/{filename}'
        if data_dir.endswith(('.mp3','.wav','.flac')):
          if filename_filter(**filename_parse(filename)):
            if data_fidelity_check(data_dir):
              dir_list.append(data_dir)
    output_file_lists.append(dir_list)
  return output_file_lists

# Pick up desired directory.
def filename_filter(**kwargs) -> bool:
  if DATASET_TYPE == 'PLAY':
    return (kwargs['y'] != -1)
  elif DATASET_TYPE == 'WAVETABLE':
    return (kwargs['y'] != -1)


# file directory -> args
def filename_parse(directory):
  if DATASET_TYPE == 'PLAY':
    pattern = re.compile(r"([A-Za-z]+)_([A-Za-z]+)_([0-9]+)-([0-9]+)-([0-9]+)\.wav", re.IGNORECASE)
    match = re.search(pattern, directory)
    try: y = WAVEFORM_NAMES.index(match.group(1))
    except: y = -1
    method = nsynth_method_dictionary[match.group(2)]
    pitch, velocity = int(match.group(4)), int(match.group(5))
    args = {'y': y,
            'pitch': pitch,
            'amp': velocity,}
    
  elif DATASET_TYPE == 'WAVETABLE':
    pattern = re.compile(r"(.+)_([0-9]+)\.wav", re.IGNORECASE)
    match = re.search(pattern, directory)
    try: y = WAVEFORM_NAMES.index(match.group(1))
    except: y = -1
    pos = int(match.group(2))
    args = {'y': y,
            'pos': pos,
            } # should be correspond to
    
  return args

def data_fidelity_check(data_dir):
  # put off silence
  x, f_s = torchaudio.load(data_dir)
  ind = min(48000 - RAW_LEN - 1, torch.argmax(x).item())
  wave = x[::, ind:ind + RAW_LEN]
  return torch.sum(wave.pow(2)) > 1e-5

class DatasetBuilder(Dataset):
  
  # this is a child class of torch.utils.data.Dataset.
  # this class is the definition of dataset.

  def __init__(self, file_list=None, path=None):
    self.path = path
    self.file_list = file_list

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, index: int) -> list: #retrieve the transformed waveform of the selected index.
    file_dir = self.file_list[index]
    wave, f_s = torchaudio.load(file_dir)
    wave = wave.squeeze()
    filename = file_dir[file_dir.rfind('/') + 1:]
    return self.transform(filename, wave, f_s) #self.transform(wave, filename)

  def transform(self, filename, x, f_s):

    if DATASET_TYPE == 'WAVETABLE':
      # normalise
      x = x[::2] #2048 -> 1024
      amp = (torch.sum(x.pow(2), dim=-1).sqrt().unsqueeze(-1))
      x /= amp
      x *= NORMALISED_ENERGY
      y, pos = filename_parse(filename).values() # should be correspond to
      return x, y, amp, pos

    elif DATASET_TYPE == 'PLAY':
      y, method, pitch, velocity = filename_parse(filename).values()
      ind = min(48000-RAW_LEN-1,torch.argmax(x).item())
      wave = x[ind:ind+RAW_LEN]
      return wave.float(), y, pitch, velocity

def data_build(datasets, folds, BS, loaderonly=True, num_workers=NUM_WORKERS) -> tuple:
  '''
  fold : function
  -------------
  1    : trains
  0    : test
  -1   : valid
  X    : pass
  else : n-fold
  -------------
  '''
  train_databuilders, test_databuilders, val_databuilders, train_loaders, test_loaders, val_loaders = [], [], [], [], [], []
  for ind, dataset in enumerate(datasets):
    if folds[ind] == 'X': continue
    elif folds[ind] == 1: #train
      current_databuilder = DatasetBuilder(file_list=dataset, train=1)
      train_databuilders.append(current_databuilder)
      train_loaders.append(DataLoader(dataset=current_databuilder, batch_size=BS, drop_last=True, shuffle=True, num_workers=num_workers))
      
    elif folds[ind] == 0: #test
      current_databuilder = DatasetBuilder(file_list=dataset)
      test_databuilders.append(current_databuilder)
      test_loaders.append(DataLoader(dataset=current_databuilder, batch_size=BS, drop_last=True, shuffle=False, num_workers=num_workers))
      
    elif folds[ind] == -1: #valid
      current_databuilder = DatasetBuilder(file_list=dataset)
      val_databuilders.append(current_databuilder)
      val_loaders.append(DataLoader(dataset=current_databuilder, batch_size=BS, drop_last=True, shuffle=False, num_workers=num_workers))
      
    else:
      assert (type(folds[ind]) is int) and (folds[ind] >= 2)
      train_size = int((folds[ind]/10) * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

      train_databuilder = DatasetBuilder(file_list=train_dataset)
      train_databuilders.append(train_databuilder)
      print(len(train_databuilder))
      train_loaders.append(DataLoader(dataset=train_databuilder, batch_size=BS, drop_last=True, shuffle=True, num_workers=num_workers))

      test_databuilder = DatasetBuilder(file_list=test_dataset)
      test_databuilders.append(test_databuilder)
      print(len(test_databuilder))
      test_loaders.append(DataLoader(dataset=test_databuilder, batch_size=BS, drop_last=True, shuffle=False, num_workers=num_workers))

  if loaderonly: return train_loaders, test_loaders, val_loaders
  else: return train_databuilders, test_databuilders, val_databuilders, train_loaders, test_loaders, val_loaders