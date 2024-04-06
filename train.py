import sys
from module import *
from funcs import *
from config import *
import os
import shutil

torch.manual_seed(0)
torch.cuda.empty_cache()

train_loaders, test_loaders, val_loaders = data_build(
    DATASETS,
    [-2], #1:train 0:test -1:valid, X:pass, else:n-fold
    BS=BS,
    loaderonly=True
    )

def get_parent_directory(directory):
    # 마지막 슬래시('/') 이후의 인덱스 찾기
    last_slash_index = directory.rfind('/')
    
    # 마지막 슬래시('/') 이후의 부분을 제외한 문자열 반환
    parent_directory = directory[:last_slash_index]

    return parent_directory

if __name__ == '__main__':
    for SET in [0,1,2,3,4]:
        #SETTINGS
        if WANDB == 'TRAIN':
            wandb.init(
                # set the wandb project where this run will be logged
                project = EXP_NAME,
                name = f'S{TINY}_PL{LEARN_PRIORS}_SET{SET}',
                #track hyperparameters and run metadata
                config={
                "SPECTRAL_LOSS_COEF": SPECTRAL_LOSS_COEF,
                "WAVEFORM_LOSS_COEF": WAVEFORM_LOSS_COEF,
                "SEMANTIC_LOSS_COEF": SEMANTIC_LOSS_COEF,
                "KL_LOSS_COEF": KL_LOSS_COEF,
                "PHASE_LOSS_COEF": PHASE_LOSS_COEF,
                "NOISE_LOSS_COEF": NOISE_LOSS_COEF,
                "LR": LR,
                "SEED": SEED,
                "WAVEFORM_LOSS_MULTIPLIER": WAVEFORM_LOSS_MULTIPLIER,
                "WAVEFORM_LOSS_DECREASE_RATE": WAVEFORM_LOSS_DECREASE_RATE,
                }
            )
        
        if CKPT_LOAD:
            wavespace = Wavespace().load_from_checkpoint(CKPT_TEST).to(DEVICE)
        else:
            wavespace = Wavespace().to(DEVICE)

        print(f'###{AB_D} {AB_L} {AB_S}###SET{SET}')
    # Train.
        trainer = pl.Trainer(max_epochs=EPOCH,
                            resume_from_checkpoint=CKPT_TEST if CKPT_LOAD else None,
                            accelerator='gpu',
                            devices=[GPU_NUM],
                            enable_progress_bar=True,
                            default_root_dir=PARENT_PATH / f"wss/log",
                            )
        trainer.fit(wavespace,
                    train_dataloaders=train_loaders[SET],
                    )
        if WANDB:
            wandb.run.name = f'S{TINY}_PL{LEARN_PRIORS}_SET{SET}'
            CKPT_PATH = PARENT_PATH / f'wss/ckpt/{EXP_NAME}_{wandb.run.name}.pth'
        else:
            while True:
                k = random.randint(0,2**12)
                if k % 5 != 0: break
                CKPT_PATH = PARENT_PATH / f'wss/ckpt/{EXP_NAME}_{k}.pth'
        trainer.save_checkpoint(CKPT_PATH)
        print(f'CKPT saved to {CKPT_PATH}')

##### Test
    #trainer.test(wavespace, test_loaders[0])
        wandb_dir_to_delete = None
        # wandb.init() this should have been ran already
        wandb_dir_to_delete = wandb.run.dir
        wandb.finish()
        if wandb_dir_to_delete is not None:
            shutil.rmtree(get_parent_directory(wandb_dir_to_delete))
            print(f'{wandb_dir_to_delete=} {type(wandb_dir_to_delete)=}')