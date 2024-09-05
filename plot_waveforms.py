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

    # Plot.
    for C, Q in [(3, 6), (3, 9), (4, 1), (2, 1), (5, 15), (5, 16), (8, 6), (10, 6)]:
        r1=(0,9) # Range 1
        r2=(0,9) # Range 2
        num_x = 5 # Number of Plots 1
        num_y = 5 # Numbrer of Plots 2
        z1_range=np.linspace(r1[0],r1[1],num_x)
        z2_range=np.linspace(r2[0],r2[1],num_y)
        query_name = WAVEFORM_NAMES[Q]
        query_num = '126'
        query = f'{query_name}_{query_num}' #Condition
        query_path = f'/workspace/wss/SerumDataset/{query}.wav'
        print(query_path)
        with torch.no_grad():
            x, y = db[db.file_list.index(filepath)]
            x = x.unsqueeze(0).to(DEVICE)
            w, _, _, _ = wavespace.encoder(x)
            features = get_semantic_conditions(x)
            fig, axes = plt.subplots(len(z1_range), len(z2_range), figsize=(80, 48))
            for n, z2 in enumerate(z2_range):
                for m, z1 in enumerate(z1_range):
                    #z = torch.tensor([0, 0]).float().unsqueeze(0).to(DEVICE)
                    #w = torch.zeros_like(w).to(DEVICE)
                    w[0,2*C] = z1
                    #w[0,2*C+1] = z2
                    features[0,1] = z2/3
                    w_s = torch.concatenate((w,features), dim=-1)
                    x_hat = wavespace.decoder(w_s).to('cpu')
                    out = x_hat
                    out = out.squeeze()
                    out = torch.concatenate((torch.zeros(1),out))
                    #dc_x_hat = torch.concatenate((torch.zeros(1), x_hat), dim=0)
                    axes[m,n].plot(out, linewidth=6, c='black')
                    axes[m,n].set_title(f"w = {z1}, {z2}")
                    axes[m,n].set_xticks([])
                    axes[m,n].set_yticks([])
                    axes[m,n].grid(True)
                    print(m,n)
            plt.tight_layout()
            folder_name = f'./fig/Z/{CKPT_NAME}/C{WAVEFORM_NAMES[C]}/Q{query}'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            directory = folder_name + f'/{r1}{r2}{num_x}x{num_y}_2.png'
            plt.savefig(directory)
            print(f'plot saved to {directory}')