import torch
import torch.nn as nn
from pathlib import Path
from . import dream_models
from nextFrag.config import get_project_root, DATASET_CONFIG

def init_model(dataset: str, arch: str) -> nn.Module:
    seqsize=DATASET_CONFIG[dataset]['seqsize']
    num_channels=DATASET_CONFIG[dataset]['in_channels']
    match arch:
        case 'rnn':
            return dream_models.DREAM_RNN(in_channels=num_channels,final_block=dataset, seqsize=seqsize)
        case 'cnn':
            return dream_models.DREAM_CNN(in_channels=num_channels,final_block=dataset,seqsize=seqsize)
        case 'attn':
            return dream_models.DREAM_ATTN(in_channels=num_channels,final_block=dataset,seqsize=seqsize)
        case _:
            raise ValueError("Model architecture must be 'cnn','rnn', or 'attn'")

def load_model(
    dataset: str,
    arch: str,
    path: str | Path = None, 
    al_strategy: str = None,
    seed: int = 42,
    round_num: int = None,
) -> nn.Module:
    '''
    expects either a path to a model.pth file OR 
    constructs path with round_num, al_strategy, seed
    '''
    model = init_model(dataset=dataset, arch=arch)
    if path is not None:
        filepath=path
    else: 
        filepath = get_project_root() / dataset / f'round_{round_num}' / al_strategy / f'{arch}_{seed}' / 'model' / 'model_best.pth'
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(filepath, weights_only=True))
    else: # cpu
        model.load_state_dict(torch.load(filepath, weights_only=False,map_location=torch.device('cpu')))
    return model
