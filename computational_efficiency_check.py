from module import *
from funcs import *
from config import *
import torch.nn.functional as F
from module.dataset import DatasetBuilder
from torch.fft import rfft as fft
import time
from thop import profile, clever_format

if __name__ == '__main__':
    D = torch.zeros(5,8).to(DEVICE)
    _,test_databuilders,_,_,_,_ = data_build(
    DATASETS,
    [-2], #1:train 0:test -1:valid, X:pass, else:n-fold
    BS=BS,
    loaderonly=False
    )

    for SET in range(1):
        print(SET)
        input_data = torch.randn(1, 41).to(DEVICE)
        wavespace = Wavespace().to(DEVICE) #.load_from_checkpoint(CKPT_TEST).to(DEVICE)
        wavespace.eval()
        db = test_databuilders[SET]
        num_of_data = len(db)
        
        with torch.no_grad():
            macs, params = profile(wavespace.decoder, inputs=(input_data,))
            flops = macs * 2  # FLOPs = MACs * 2
            macs, params = clever_format([macs, params], "%.3f")

            print(f"MACs: {macs}")
            print(f"FLOPs: {flops}")    