from module import *
from funcs import *
from config import *
from module.dataset import DatasetBuilder
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ##LOAD
    load_ckpt = torch.load(CKPT_TEST)
    loaded_model_state_dict = load_ckpt['state_dict']
    #loaded_optimizer_state_dict = load_ckpt['optimizer_state_dict']

    wavespace = Wavespace()
    wavespace.load_state_dict(loaded_model_state_dict)
    wavespace = wavespace.to(DEVICE) #after train/test, the model automatically set to CPU
    wavespace.eval()
    #optimizer = optim.Adam(wavespace.parameters(), lr=0.001)
    #optimizer.load_state_dict(loaded_optimizer_state_dict)
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
        print(f'Query:{query}, Condition:{WAVEFORM_NAMES[C]}')

        query_path = PARENT_PATH / f'wss/SerumDataset/{query}.wav'
        with torch.no_grad():
            i_tensor = torch.tensor(1j, dtype=torch.complex64)
            indices = [index for index, value in enumerate(db.file_list) if value == query_path]
            i = indices[0]
            datum = list(db[i])
            for j in range(len(datum)):
                if isinstance(datum[j], int):
                    datum[j] = torch.Tensor([datum[j]]).to(torch.int64).to(DEVICE)
                else: datum[j] = datum[j].reshape(1,-1).to(DEVICE)

            x, y, amp, pos = tuple(datum)
            mu_w, logvar_w = wavespace.encoder(x) #x, x_hat, mu_w, logvar_w, y
            w = mu_w

            fig, axes = plt.subplots(len(z1_range), len(z2_range), figsize=(80, 48))
            for n, z2 in enumerate(z2_range):
                for m, z1 in enumerate(z1_range):
                    #z = torch.tensor([0, 0]).float().unsqueeze(0).to(DEVICE)
                    #w = torch.zeros_like(w).to(DEVICE)
                    w[0,2*C] = z1
                    w[0,2*C+1] = z2
                    #pos = torch.tensor([[z2]]).float().to(DEVICE)
                    #print(f'W: {w}')
                    #print(f'pos: {pos}')
                    #f0 = f0.to(DEVICE)
                    #amp = amp.to(DEVICE)
                    #amp = torch.tensor([z2]).float().unsqueeze(0).to(DEVICE)
                    x_hat = wavespace.decoder(w).to('cpu')
                    out = x_hat #* amp.to('cpu')
                    out = out.squeeze()
                # axes[m,n].plot(x_hat.to('cpu'), linewidth=1)
                    out = torch.concatenate((torch.zeros(1),out))
                    #dc_x_hat = torch.concatenate((torch.zeros(1), x_hat), dim=0)
                    axes[m,n].plot(out, linewidth=6, c='black')
                    axes[m,n].set_title(f"w = {z1}, {z2}")
                    axes[m,n].set_xticks([])
                    axes[m,n].set_yticks([])
                    axes[m,n].grid(True)
            plt.tight_layout()
            folder_name = f'./fig/Z/{CKPT_NAME}/C{WAVEFORM_NAMES[C]}/Q{query}'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            directory = folder_name + f'/{r1}{r2}{num_x}x{num_y}.png'
            plt.savefig(directory)
            print(f'plot saved to {directory}')