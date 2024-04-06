from module import *
from funcs import *
from config import *
from module.dataset import DatasetBuilder
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ###CHECKPOINT LOAD###
    wavespace = Wavespace().load_from_checkpoint(CKPT_TEST).to(DEVICE)
    wavespace.eval()
    db = DatasetBuilder(file_list=DATASETS[0])
    with torch.no_grad():
        i_tensor = torch.tensor(1j, dtype=torch.complex64).to(DEVICE)
        for C in [2,3]:
            fig, axes = plt.subplots(16,16,figsize=(80, 48))
            print(f'progress: {C}/17')
            for m in range(4):
                for n in range(4):
                    num = str(16*m+n+1+10**3)[1:] #str(16*m+n+1+10**3)[1:]
                    c = f'{WAVEFORM_NAMES[C]}_{num}' #Condition
                    filepath = f'/workspace/wss/SerumDataset/{c}.wav' #f'/workspace/wss/WaveEditDataset/processed_upscaled/{WAVEFORM_NAMES[C]}/{c}.wav'
                    x, y = db[db.file_list.index(filepath)]
                    x = x.unsqueeze(0).to(DEVICE)
                    w, _, x, _ = wavespace.encoder(x)
                    axes[m,n].plot(x.squeeze().cpu())
                    w = torch.cat((w,get_semantic_conditions(x)), dim=-1)
                    x_hat, amp = wavespace.decoder(w)
                    x_hat = x_hat.squeeze()
                    axes[m,n].plot(x_hat.cpu())
                    print(m,n)
            plt.tight_layout()
            folder_name = f'./fig/R/{CKPT_NAME}'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            directory = folder_name + f'/{WAVEFORM_NAMES[C]}.png'
            plt.savefig(directory)
            print(f'plot saved to {directory}')