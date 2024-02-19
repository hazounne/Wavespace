from funcs import *
from config import *
import os
import torch
import torchaudio
import re
from torch.utils.data import Dataset, DataLoader
from torch.fft import rfft

'''
[order of data preparation]
data_folders
-> collapse (DATASETS from module.__init__)
-> DatasetBuilder
-> loaders(assigned in experiment): concat each datum in the list
'''

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

def filename_filter(**kwargs) -> bool:
  # filters that pick up desired directory
  if DATASET_TYPE == 'PLAY':
    return (kwargs['y'] != -1)
  elif DATASET_TYPE == 'WAVETABLE':
    return (kwargs['y'] != -1)


def filename_parse(directory):
  # file directory -> args
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

  def __init__(self, file_list=None, train=0, path=None, fold=None): #fold: train, test fold
    self.train = train #if it's a training dataset
    self.path = path
    if fold:
      if self.train:
        self.file_list = file_list[:(fold-1)*len(file_list)//fold]
      else:
        self.file_list = file_list[(fold-1)*len(file_list)//fold:]
    else: self.file_list = file_list

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
      # x = torch.clamp(x, min=0, max=1)

      y, pos = filename_parse(filename).values() # should be correspond to
      #f0 = torch.ones(1) * 440 #arbitrary
      features = dco_extractFeatures(x, tile_num=6)
      return x, y, amp, pos, features

    elif DATASET_TYPE == 'PLAY':
      y, method, pitch, velocity = filename_parse(filename).values()
      ind = min(48000-RAW_LEN-1,torch.argmax(x).item())
      wave = x[ind:ind+RAW_LEN]
      return wave.float(), y, pitch, velocity

# def filename_to_y(filename: str, waveform_names: list) -> int:
#   for i, name in enumerate(waveform_names):
#     if name in filename:
#       return i
#   return N_CONDS-1 #no cond -> assign to the last one

def data_build(li, folds, BS, loaderonly=True, num_workers=NUM_WORKERS) -> tuple:
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
  train_li, test_li, val_li, train_loaders, test_loaders, val_loaders = [], [], [], [], [], []
  for ind, dataset in enumerate(li):
    if folds[ind] == 'X': continue
    elif folds[ind] == 1: #train
      c = DatasetBuilder(file_list=dataset, train=1)
      train_li.append(c)
      train_loaders.append(DataLoader(dataset=c, batch_size=BS, drop_last=True, shuffle=True, num_workers=num_workers))
      
    elif folds[ind] == 0: #test
      c = DatasetBuilder(file_list=dataset)
      test_li.append(c)
      test_loaders.append(DataLoader(dataset=c, batch_size=BS, drop_last=True, shuffle=False, num_workers=num_workers))
      
    elif folds[ind] == -1: #valid
      c = DatasetBuilder(file_list=dataset)
      val_li.append(c)
      val_loaders.append(DataLoader(dataset=c, batch_size=BS, drop_last=True, shuffle=False, num_workers=num_workers))
      
    else:
      assert (type(folds[ind]) is int) and (folds[ind] >= 2)

      c = DatasetBuilder(file_list=dataset, train=1, fold=folds[ind])
      train_li.append(c)
      train_loaders.append(DataLoader(dataset=c, batch_size=BS, drop_last=True, shuffle=True, num_workers=num_workers))
      c = DatasetBuilder(file_list=dataset, fold=folds[ind])
      val_li.append(c)
      val_loaders.append(DataLoader(dataset=c, batch_size=BS, drop_last=True, shuffle=False, num_workers=num_workers))

  if loaderonly: return train_loaders, test_loaders, val_loaders
  else: return train_li, test_li, val_li, train_loaders, test_loaders, val_loaders