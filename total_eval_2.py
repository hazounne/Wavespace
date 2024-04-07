from module import *
from funcs import *
from config import *
import torch.nn.functional as F
from module.dataset import DatasetBuilder
from torch.fft import rfft as fft
import time

def minmax_normal(data, range=(-1,1)):
        min_val = torch.min(data)
        max_val = torch.max(data)
        normalized_data = abs(range[0]-range[1]) * ((data - min_val) / (max_val - min_val) + ((range[0] + range[1]) / 2 - 0.5))
        return normalized_data

if __name__ == '__main__':
    D = torch.zeros(5,8).to(DEVICE) #MAE(0) MSE(1) cMAE(2:7) KD(7)
    T = 0
    _,test_databuilders,_,_,_,_ = data_build(
    DATASETS,
    [-2], #1:train 0:test -1:valid, X:pass, else:n-fold
    BS=BS,
    loaderonly=False
    )
    print(f'TEST INITIALISING::: S{TINY}_PL{LEARN_PRIORS}')
    repeat = 1
    for rep in range(repeat):
        print(rep)
        for SET in range(5):
            print(SET)
            if WAVEFORMS == waveedit:
                print('WAVEEDIT')
                CKPT_NAME = f'{EXP_NAME}_S{TINY}_PL{LEARN_PRIORS}_SET{SET}'
            elif WAVEFORMS == serum_sub2_B:
                print('SERUM')
                CKPT_NAME = f'{EXP_NAME}_S{TINY}_PL{LEARN_PRIORS}_SET{SET}'
            CKPT_TEST = PARENT_PATH / f'wss/ckpt/{CKPT_NAME}.pth'
            wavespace = Wavespace().load_from_checkpoint(CKPT_TEST).to(DEVICE)
            wavespace.eval()
            db = test_databuilders[SET]
            num_of_data = len(db)
            
            with torch.no_grad():
                for i in range(num_of_data):
                    x, y = db[i]
                    x = x.unsqueeze(0).to(DEVICE)
                    A, t = wavespace((x,y),return_decoder_time=True)

                    D[SET,:] += wavespace.loss_values(*A).squeeze()
                    T += t
    print(D.div(num_of_data))
    avgtime = T/(num_of_data * repeat)
    print(f'avgtime={avgtime}')
    print(f'RTF={(1024/48000)/avgtime}')
    print(f'Result:::{D.div(num_of_data).mean(dim=0)}')
    #torch.save(wavespace.state_dict(), f'/workspace/wss/ckpt/pt/{CKPT_NAME}.pt')
                
