from module import *
from funcs import *
from config import *
from module.dataset import DatasetBuilder

if __name__ == '__main__':
    wavespace = Wavespace().to(DEVICE)
    ###CHECKPOINT LOAD###
    load_ckpt = torch.load(CKPT_TEST)
    loaded_model_state_dict = load_ckpt['state_dict']
    if STAGE == 1:
        new_state_dict = wavespace.state_dict()
        for key in new_state_dict.keys():
            if not 'discriminator' in key:
                new_state_dict[key] = loaded_model_state_dict[key]
        wavespace.load_state_dict(new_state_dict)
    elif STAGE == 2: wavespace.load_state_dict(loaded_model_state_dict)

    print(f"checkpoint_loaded:{CKPT_TEST}")
    if STAGE == 2:
        for param in wavespace.encoder.parameters():
            param.requires_grad = False
    wavespace.eval()
    #optimizer = optim.Adam(wavespace.parameters(), lr=0.001)
    #optimizer.load_state_dict(loaded_optimizer_state_dict)

    # Plot.
    db = DatasetBuilder(file_list=DATASETS[0])
    div = 18*256
    wMAE = 0
    wMSE = 0
    sMAE = 0
    sMSE = 0
    with torch.no_grad():
        i_tensor = torch.tensor(1j, dtype=torch.complex64).to(DEVICE)
        for C in range(18):
            print(f'progress: {C}/17')
            for n in range(16):
                num = str(n+1+10**3)[1:]
                c = f'{WAVEFORM_NAMES[C]}_{num}' #Condition
                #print(c)
                #x_path = PARENT_PATH / f'wss/SerumDataset/{c}.wav'
                x_path = f'/workspace/wss/SerumDataset/{c}.wav'
                
                indices = [index for index, value in enumerate(db.file_list) if value == x_path]
                if not indices:
                    print(n)
                    assert indices
                i = indices[0]
                datum = list(db[i])
                x = datum[0]
                for j in range(len(datum)):
                    if isinstance(datum[j], int):
                        datum[j] = torch.Tensor([datum[j]]).to(torch.int64).to(DEVICE)
                    else: datum[j] = datum[j].reshape(1,-1).to(DEVICE)
                x, y, amp, pos, features = tuple(datum)
                mu_w, logvar_w = wavespace.encoder(x) #x, x_hat, mu_w, logvar_w, y
                w = mu_w
                w = torch.concatenate((w,features.to(DEVICE)), dim=-1)
                x_hat = wavespace.decoder(w)
                x_hat = x_hat.squeeze()
                wMAE += torch.sum((x - x_hat).abs()).item()/div
                wMSE += torch.sum((x - x_hat).pow(2)).item()/div
                x_s = fft.fft(x).abs()
                x_hat_s = fft.fft(x_hat).abs()
                sMAE += torch.sum((x_s - x_hat_s).abs()).item()/div
                sMSE += torch.sum((x_s - x_hat_s).pow(2)).item()/div
        print(f'wMAE = {wMAE}')
        print(f'wMSE = {wMSE}')
        print(f'sMAE = {sMAE}')
        print(f'sMSE = {sMSE}')