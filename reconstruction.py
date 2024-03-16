from module import *
from funcs import *
from config import *
from module.dataset import DatasetBuilder
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ###CHECKPOINT LOAD###
    wavespace = Wavespace().load_from_checkpoint(CKPT_TEST).to(DEVICE)
    wavespace.eval()
    #optimizer = optim.Adam(wavespace.parameters(), lr=0.001)
    #optimizer.load_state_dict(loaded_optimizer_state_dict)

    # Plot.
    db = DatasetBuilder(file_list=DATASETS[0])
    with torch.no_grad():
        i_tensor = torch.tensor(1j, dtype=torch.complex64).to(DEVICE)
        for C in [0,1]:
            fig, axes = plt.subplots(16,16,figsize=(80, 48))
            print(f'progress: {C}/17')
            for m in range(16):
                for n in range(16):
                    num = str(16*m+n+1+10**4)[1:] #str(16*m+n+1+10**3)[1:]
                    c = f'{WAVEFORM_NAMES[C]}_{num}' #Condition
                    #print(c)
                    #x_path = PARENT_PATH / f'wss/SerumDataset/{c}.wav'
                    x_path = f'/workspace/wss/WaveEditDataset/processed_upscaled/{WAVEFORM_NAMES[C]}/{c}.wav'#f'/workspace/wss/SerumDataset/{c}.wav'
                    
                    indices = [index for index, value in enumerate(db.file_list) if value == x_path]
                    assert indices
                    i = indices[0]
                    datum = list(db[i])
                    x = datum[0]
                    for j in range(len(datum)):
                        if isinstance(datum[j], int):
                            datum[j] = torch.Tensor([datum[j]]).to(torch.int64).to(DEVICE)
                        else: datum[j] = datum[j].reshape(1,-1).to(DEVICE)
                    x, y = tuple(datum)
                    mu_w, logvar_w, x, x_spec = wavespace.encoder(x) #x, x_hat, mu_w, logvar_w, y
                    w = mu_w
                    w = torch.cat((w,get_semantic_conditions(x)), dim=-1)
                    x_hat, amp = wavespace.decoder(w)
                    x_hat = x_hat.squeeze()
                    axes[m,n].plot(x.cpu())
                    axes[m,n].plot(x_hat.cpu())
            plt.tight_layout()
            folder_name = f'./fig/R/{CKPT_NAME}'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            directory = folder_name + f'/{WAVEFORM_NAMES[C]}.png'
            plt.savefig(directory)
            print(f'plot saved to {directory}')