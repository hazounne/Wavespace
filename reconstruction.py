from module import *
from funcs import *
from config import *
from module.dataset import DatasetBuilder
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ##LOAD
    load_ckpt = torch.load(CKPT_TEST)
    loaded_model_state_dict = load_ckpt['state_dict']
    # loaded_optimizer_state_dict = load_ckpt['optimizer_state_dict']

    wavespace = Wavespace()
    wavespace.load_state_dict(loaded_model_state_dict)
    wavespace = wavespace.to(DEVICE) #after train/test, the model automatically set to CPU
    wavespace.eval()
    #optimizer = optim.Adam(wavespace.parameters(), lr=0.001)
    #optimizer.load_state_dict(loaded_optimizer_state_dict)

    # Plot.
    domain = 'w'
    db = DatasetBuilder(file_list=DATASETS[0])


    with torch.no_grad():
        i_tensor = torch.tensor(1j, dtype=torch.complex64).to(DEVICE)
        for C in range(18):
            fig, axes = plt.subplots(16,16,figsize=(80, 48))
            print(f'progress: {C}/17')
            for m in range(16):
                for n in range(16):
                    num = str(16*m+n+1+10**3)[1:]
                    c = f'{WAVEFORM_NAMES[C]}_{num}' #Condition
                    #print(c)
                    #x_path = PARENT_PATH / f'wss/SerumDataset/{c}.wav'
                    x_path = f'/workspace/wss/SerumDataset/{c}.wav'
                    
                    indices = [index for index, value in enumerate(db.file_list) if value == x_path]
                    assert indices
                    i = indices[0]
                    #print(i)

                    datum = list(db[i])
                    #x = torch.exp(-datum[0])
                    x = datum[0]

                    if domain == 'w':
                        axes[m,n].plot(x.cpu())
                    else:
                        x = (x-torch.min(x))/(torch.max(x)-torch.min(x)) 
                        axes[m,n].plot(x)

                    for j in range(len(datum)):
                        if isinstance(datum[j], int):
                            datum[j] = torch.Tensor([datum[j]]).to(torch.int64).to(DEVICE)
                        else: datum[j] = datum[j].reshape(1,-1).to(DEVICE)
                    x, y, amp, pos, features = tuple(datum)
                    mu_w, logvar_w = wavespace.encoder(x) #x, x_hat, mu_w, logvar_w, y
                    w = mu_w
                    w = torch.concatenate((w,features.unsqueeze(0).to(DEVICE)), dim=-1)
                    x_hat = wavespace.decoder(w)
                    x_hat = x_hat.squeeze()
                    if domain == 'w':
                        axes[m,n].plot(x_hat.cpu())
                    else:
                        x_hat = (x_hat-torch.min(x_hat))/(torch.max(x_hat)-torch.min(x_hat)) 
                        axes[m,n].plot(x_hat.to('cpu'))
                    #axes[m,n].set_title(f"w = {w_1}, {w_2}")
                    axes[m,n].set_xticks([])
                    axes[m,n].set_yticks([])
                    axes[m,n].grid(True)
            #print(f'w={w.tolist()}')
            plt.tight_layout()
            directory = f'wss/fig/R_{CKPT_NAME}_{c}.png'
            plt.savefig(directory)
            plt.clf()
            print(f'plot saved to {directory}')