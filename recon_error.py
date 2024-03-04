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

    train_databuilders, test_databuilders, _,_,_,_ = data_build(
    DATASETS,
    [9], #1:train 0:test -1:valid, X:pass, else:n-fold
    BS=BS,
    loaderonly=False
    )

    db = train_databuilders[0]
    num_of_data = len(db)
    wMAE = 0
    wMSE = 0
    sMAE = 0
    sMSE = 0
    with torch.no_grad():
        i_tensor = torch.tensor(1j, dtype=torch.complex64).to(DEVICE)
        for C in range(18):
            print(f'progress: {C}/17')
            for i in range(num_of_data):
                x = db[i]
                x = x[0].unsqueeze(0).to(DEVICE)
                mu_w, logvar_w = wavespace.encoder(x) #x, x_hat, mu_w, logvar_w, y
                w = mu_w
                w = torch.concatenate((w,get_semantic_conditions(x)), dim=-1)
                x_hat = wavespace.decoder(w)
                x_hat = x_hat.squeeze()
                wMAE += torch.sum((x - x_hat).abs()).item()/num_of_data
                wMSE += torch.sum((x - x_hat).pow(2)).item()/num_of_data
                x_s = fft.fft(x).abs()
                x_hat_s = fft.fft(x_hat).abs()
                sMAE += torch.sum((x_s - x_hat_s).abs()).item()/num_of_data
                sMSE += torch.sum((x_s - x_hat_s).pow(2)).item()/num_of_data
        print(f'wMAE = {wMAE}')
        print(f'wMSE = {wMSE}')
        print(f'sMAE = {sMAE}')
        print(f'sMSE = {sMSE}')