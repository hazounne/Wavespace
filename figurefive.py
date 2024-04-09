from module import *
from funcs import *
from config import *
from module.dataset import DatasetBuilder
import matplotlib.pyplot as plt

if __name__ == '__main__':
    wavespace = Wavespace.load_from_checkpoint(CKPT_TEST).to(DEVICE)
    wavespace.eval()
    db = DatasetBuilder(file_list=DATASETS[0])
    with torch.no_grad():
        ### Custom
        grid_number = 5
        ###
        wavetable_A_w = []
        wavetable_B_w = []
        fig, axes = plt.subplots(grid_number,grid_number,figsize=(80, 48))
        #Draw the plot.
        for m in range(grid_number):
            for n in range(grid_number):
                w = torch.tensor([0,0,5,5,0,0,0,0]+[0,0]*14).to(DEVICE)
                sc = torch.zeros(5).to(DEVICE)
                if m != 4:
                    sc[m] = (n+0.2)/grid_number
                else: sc[m] = n*(np.pi)/2 - np.pi
                w_s = torch.cat((w,sc),dim=0)
                x_hat, _ = wavespace.decoder(w_s.unsqueeze(0))
                x_hat = x_hat.squeeze()
                axes[m,n].plot(x_hat.cpu(), c='black', linewidth=3)
                print(m,n)
                axes[m,n].set_xticks([])
                axes[m,n].set_yticks([])
                axes[m,n].grid(False)
                axes[m,n].spines['top'].set_visible(False)
                axes[m,n].spines['right'].set_visible(False)
                axes[m,n].spines['left'].set_visible(False)
                axes[m,n].spines['bottom'].set_visible(False)
        plt.tight_layout()
        folder_name = f'./fig/R/{CKPT_NAME}'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        directory = folder_name + '/FIG5.png'
        plt.savefig(directory)
        print(f'plot saved to {directory}')