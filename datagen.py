import pickle
from module import *
from funcs import *
from config import *
from module.dataset import DatasetBuilder

if __name__ == '__main__':
    db = DatasetBuilder(file_list=DATASETS[0])
    for i in db:
        