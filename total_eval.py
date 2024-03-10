from module import *
from funcs import *
from config import *
import torch.nn.functional as F
from module.dataset import DatasetBuilder
import time

if __name__ == '__main__':
    wavespace = Wavespace().load_from_checkpoint(CKPT_TEST).to(DEVICE)
    wavespace.eval()
    train_databuilders, test_databuilders, _,_,_,_ = data_build(
    DATASETS,
    [9], #1:train 0:test -1:valid, X:pass, else:n-fold
    BS=BS,
    loaderonly=False
    )

    db = test_databuilders[0]
    num_of_data = len(db)
    wMAE = 0
    wMSE = 0
    sMAE = 0
    sMSE = 0
    TIME = 0
    def minmax_normal(data, range=(-1,1)):
        min_val = torch.min(data)
        max_val = torch.max(data)
        normalized_data = abs(range[0]-range[1]) * ((data - min_val) / (max_val - min_val) + ((range[0] + range[1]) / 2 - 0.5))
        return normalized_data

    with torch.no_grad():
        i_tensor = torch.tensor(1j, dtype=torch.complex64).to(DEVICE)
        for i in range(num_of_data):
            x = db[i]
            x = x[0].unsqueeze(0).to(DEVICE)
            start = time.time() # 시작
            mu_w, logvar_w = wavespace.encoder(x) #x, x_hat, mu_w, logvar_w, y
            w = mu_w
            w = torch.concatenate((w,get_semantic_conditions(x)), dim=-1)
            x_hat = wavespace.decoder(w)
            end = time.time() # 시작
            x = minmax_normal(x.squeeze(0))
            x_hat = minmax_normal(x_hat.squeeze(0))
            TIME += end-start
            wMAE += torch.mean((x - x_hat).abs()).item()
            wMSE += torch.mean((x - x_hat).pow(2)).item()
            x_s = fft.rfft(x).abs()
            x_hat_s = fft.rfft(x_hat).abs()
            sMAE += torch.mean((x_s - x_hat_s).abs()).item()
            sMSE += torch.mean((x_s - x_hat_s).pow(2)).item()
        
        print(f'wMAE = {wMAE/(num_of_data)}')
        print(f'wMSE = {wMSE/(num_of_data)}')
        print(f'sMAE = {sMAE/(num_of_data)}')
        print(f'sMSE = {sMSE/(num_of_data)}')
        print(f'avgTIME = {TIME/(num_of_data)}')