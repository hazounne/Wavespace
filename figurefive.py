from module import *
from funcs import *
from config import *
from module.dataset import DatasetBuilder
import matplotlib.pyplot as plt

if __name__ == '__main__':
    wavespace = Wavespace().load_from_checkpoint(CKPT_TEST).to(DEVICE)
    wavespace.eval()
    db = DatasetBuilder(file_list=DATASETS[0])
    with torch.no_grad():
        ### Custom
        C = [4,5]
        grid_number = 7
        ###
        wavetable_A_w = []
        wavetable_B_w = []
        fig, axes = plt.subplots(grid_number,grid_number,figsize=(80, 48))
        for i in range(2):
            for j in range(grid_number):
                num = str((j+1)*(256//grid_number)+10**3)[1:] #10**4
                filename = f'{WAVEFORM_NAMES[C[i]]}_{num}'
                filepath = f'/workspace/wss/SerumDataset/{filename}.wav'  #filepath = f'/workspace/wss/WaveEditDataset/processed_upscaled/{WAVEFORM_NAMES[C[i]]}/{filename}.wav'
                x, y = db[db.file_list.index(filepath)]
                x = x.unsqueeze(0).to(DEVICE)
                w, _, x, _ = wavespace.encoder(x)
                w = torch.cat((w,get_semantic_conditions(x)), dim=-1)
                if i == 0: wavetable_A_w.append(w)
                elif i == 1: wavetable_B_w.append(w)
                axes[(grid_number-1)*i,j].plot(x.squeeze().cpu(), c='blue')
                print(i,j)
        
        #Draw the plot.
        for m in range(grid_number):
            for n in range(grid_number):
                w = ((grid_number-m-1)/(grid_number-1))*wavetable_A_w[n] + (m/(grid_number-1))*wavetable_B_w[n]
                x_hat, _ = wavespace.decoder(w)
                x_hat = x_hat.squeeze()
                axes[m,n].plot(x_hat.cpu(), c='black')
                print(m,n)
                axes[m,n].set_xticks([])
                axes[m,n].set_yticks([])
                axes[m,n].grid(True)
        plt.tight_layout()
        folder_name = f'./fig/R/{CKPT_NAME}'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        directory = folder_name + f'/{WAVEFORM_NAMES[C[0]]},{WAVEFORM_NAMES[C[1]]}.png'
        plt.savefig(directory)
        print(f'plot saved to {directory}')