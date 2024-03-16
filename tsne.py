from module import *
from funcs import *
from config import *
from module.dataset import DatasetBuilder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import deque as dq
from matplotlib.font_manager import FontProperties
import numpy as np
if __name__ == '__main__':
    wavespace = Wavespace().load_from_checkpoint(CKPT_TEST).to(DEVICE)
    wavespace.eval()
    wavespace.eval()
    train_databuilders, test_databuilders, _,_,_,_ = data_build(
    DATASETS,
    [9], #1:train 0:test -1:valid, X:pass, else:n-fold
    BS=BS,
    loaderonly=False
    )

    db = test_databuilders[0]

    X = dq([])
    H = dq([])
    W = dq([])
    Y = dq([])
    N = db.__len__()

    #print(db.__len__())
    for i in range(N):
        if i%100==0: print(f'{i}/{N} done.')
        datum = list(db[i])
        for j in range(len(datum)):
            if isinstance(datum[j], int):
                if j==1: Y.append(datum[j])
                datum[j] = torch.Tensor([datum[j]]).to(torch.int64).to(DEVICE)
            else: datum[j] = datum[j].reshape(1,-1).to(DEVICE)
        _, x_hat, _, _, mu_w, logvar_w, y = wavespace(tuple(datum))
        
        if DATASET_TYPE=='PLAY': _, x = overtone(datum[0], f_s=16000, n=N_OVERTONES, f_0=datum[3])
        elif DATASET_TYPE == 'WAVETABLE': x = datum[0]

        w = wavespace.sampling(mu_w, logvar_w, y)
        X.append(x.squeeze().to('cpu').tolist())
        W.append(w.squeeze().to('cpu').tolist())
        H.append(x_hat.squeeze().to('cpu').tolist())

    tsne = TSNE(n_components=2,
                  random_state=42,
                  learning_rate='auto',
                  init='random',
                  perplexity=310,
                )
    X_tsne = tsne.fit_transform(np.array(list(X)))
    W_tsne = tsne.fit_transform(np.array(list(W)))
    H_tsne = tsne.fit_transform(np.array(list(H)))
    # W_tsne = tsne.fit_transform(np.array(list(W)))

    # Separate data points by categories
    categories = np.unique(np.array(Y))

    # Create a color map for the categories
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(categories)))

    W = W_tsne
    # Create a scatter plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for i, category in enumerate(categories):
        axs[1].scatter(W[Y == category, 0],
                    W[Y == category, 1],
                    color=colors[i],
                    label=WAVEFORM_NAMES[category],
                    s=3)

    # Add labels and a title
    #plt.legend(loc='upper right', title='conditions', fontsize='small', bbox_to_anchor=(1.5, 1))

    # Create a scatter plot
    for i, category in enumerate(categories):
        axs[0].scatter(X_tsne[Y == category, 0],
                    X_tsne[Y == category, 1],
                    color=colors[i],
                    label=WAVEFORM_NAMES[category],
                    s=3
                    )

    a = 16
    import matplotlib.font_manager

    # 시스템에 설치된 폰트 목록 가져오기
    font_list = matplotlib.font_manager.findSystemFonts()

    # 폰트 목록 출력
    for font in font_list:
        print(font)
    
    # Add labels and a title
    for k in range(2):
        axs[k].set_xlabel('')
        axs[k].set_ylabel('')
        axs[k].set_xlabel('')
        axs[k].set_ylabel('')
        axs[k].set_xticks([])
        axs[k].set_yticks([])
        # axs[k].tick_params(axis='both', which='major', labelsize=a*2)
    
    font_props = FontProperties(fname='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', size=a)
    axs[0].set_title('X', fontproperties=font_props)
    axs[1].set_title('W', fontproperties=font_props)
    plt.tight_layout()
    plt.savefig(PARENT_PATH / f'wss/fig/XW_{CKPT_NAME}.png')
    plt.clf()

# #XHAT
#     # Create a scatter plot
#     for i, category in enumerate(categories):
#         plt.scatter(H_tsne[Y == category, 0],
#                     H_tsne[Y == category, 1],
#                     color=colors[i],
#                     label=WAVEFORM_NAMES[category],
#                     s=0.5
#                     )

#     # Add labels and a title
#     plt.legend(loc='upper right', title='conditions', fontsize='small', bbox_to_anchor=(1.5, 1))
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.title('X_hat')
#     plt.savefig(PARENT_PATH / f'wss/fig/H_{CKPT_NAME}.png')
#     plt.clf()