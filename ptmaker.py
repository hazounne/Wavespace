import torch
from config import *
import torch.nn as nn
from module.ptmodel import Wavespace

wavespace = Wavespace().load_from_checkpoint(CKPT_TEST).to(DEVICE)
wavespace.eval()

if __name__ == '__main__':
    with torch.no_grad():
        w = torch.randn((1,N_CONDS*2+5))*2+3
        traced_script = wavespace.to_torchscript(method="trace", example_inputs=w)
        torch.jit.save(traced_script, f'/workspace/wss/ckpt/traced_pt/{CKPT_NAME}.pt')