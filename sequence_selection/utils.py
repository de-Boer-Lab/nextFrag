import numpy as np
import pandas as pd
import torch, gc
from pathlib import Path
from typing import List, Optional
from torch import nn
from torch.utils.data import DataLoader
from nextFrag.config import get_project_root

try:
    import cupy as cp
except ImportError:
    pass

def update_train_and_pool(
    dataset: str,
    current_round: int,
    al_strategy: str,
    arch: str,
    seed: int,
    num_rounds: int=3
):
    """
    Update training and pool data between active learning rounds.
    
    Assumes the AL directory structure:
    PROJECT_ROOT/{dataset}/round_{round}/{al_strategy}/{arch}_{seed}/data/
    containing: selected.txt, train.txt, pool.txt
    
    Args:
        dataset: Dataset name ('yeast' or 'human')
        current_round: The round of selection that just finished
        al_strategy: Active learning strategy name
        arch: Model architecture name
        seed: Random seed for this run
        num_rounds: Total number of rounds in the experiment
    """
    dataset_root = get_project_root() / dataset
    
    if current_round == 1:
        prev_train = dataset_root / "round_0" / "common" / "train.txt"
        prev_pool = dataset_root / "round_0" / "common" / "pool.txt"
    else:
        prev_data_dir = (dataset_root / f"round_{current_round - 1}" / 
                         al_strategy / f"{arch}_{seed}" / "data")
        prev_train = prev_data_dir / "train.txt"
        prev_pool = prev_data_dir / "pool.txt"

    curr_data_dir = (
        dataset_root / f"round_{current_round}" / 
        al_strategy / f"{arch}_{seed}" / "data"
    )
    curr_selected = curr_data_dir / "selected.txt"
    curr_train = curr_data_dir / "train.txt"
    curr_pool = curr_data_dir / "pool.txt"
    curr_data_dir.mkdir(parents=True, exist_ok=True)

    update_train(
        prev_train=prev_train,
        selected=curr_selected,
        curr_train=curr_train
    )
    if current_round < num_rounds:
        update_pool(
            prev_pool=prev_pool,
            selected=curr_selected,
            curr_pool=curr_pool
        )
    if current_round > 1:
        prev_train.unlink(missing_ok=True)
        prev_pool.unlink(missing_ok=True)

def update_train(prev_train: str | Path,
                 selected: str | Path,
                 curr_train: str | Path):
    with open(curr_train, "w") as f:
        for in_file in [prev_train, selected]:
            with open(in_file, "r") as infile:
                f.write(infile.read())

def update_pool(prev_pool: str | Path, 
                selected: str | Path, 
                curr_pool: str | Path):
    with open(selected, "r") as f:
        selected_seqs = set(f.readlines()) 

    with open(prev_pool, "r") as f1, open(curr_pool, "w") as f2:
        for line in f1:
            if line not in selected_seqs:  
                f2.write(line)
        
def write_selections(
    result_df: pd.DataFrame,
    dataset: str,
    strategy: str,
    round_num: int,
    num_selected: int = 20_000,
    arch: str = None,
    seed: int = None,
    symlinks: Optional[List[str]] = None,
):
    arch = arch if arch is not None else "model"
    seed = seed if seed is not None else 0

    if num_selected == 20_000:
        folder_name = strategy
    else:
        folder_name = f"{strategy}_{num_selected // 1000}k"

    run_dir = (
        get_project_root() / dataset / f"round_{round_num}" / folder_name
        / f"{arch}_{seed}"
    )
    out_path = run_dir / "data" / "selected.txt"
    out_path.parent.mkdir(parents=True,exist_ok=True)
    result_df.to_csv(out_path,sep='\t', index=False,header=False,columns=[0,1])

    if symlinks:
        for name in symlinks:
            link = run_dir.parent / name
            if not link.exists():
                link.symlink_to(run_dir, target_is_directory=True)

def _forward(model: nn.Module, dream_model: bool, X: torch.Tensor):
    return model.predict(X) if dream_model else model(X)

def enable_dropout(model: nn.Module):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

class LayerInputExtractor:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.layer = layer
        self._features = None
        # register hook
        self.hook = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        # save the output of the chosen layer
        self._features = inputs[0].detach()

    def __call__(self, x):
        _ = self.model(x)
        return self._features

    def close(self):
        self.hook.remove()

def free_gpu_mem():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def free_gpu_mem_gc():
    free_gpu_mem()
    gc.collect()

def get_last_layer(model: nn.Module, 
                   dataloader: DataLoader, 
                   device: torch.device):
    model.to(device).eval()
    extractor=LayerInputExtractor(model,model.final_block)
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)
            result = extractor(X)
            result=result.reshape((result.shape[0],-1))
            half_batch = result.shape[0]//2
            yield (result[:half_batch,:]+result[half_batch:,:])/2

def last_layer_features(dataloader: DataLoader,
                        model: nn.Module) -> np.ndarray: # CURRENTLY UNUSED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    extractor=LayerInputExtractor(model,model.final_linear[0])

    results=[]
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)
            results.append(extractor(X))
    combined = torch.cat(results).cpu().numpy()
    num_samples=combined.shape[0]//2
    return (combined[:num_samples]+combined[num_samples:])/2

def distance_np(target: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.sum((target-X)**2,axis=1)

def distance_torch(target: torch.Tensor,X: torch.Tensor) -> torch.Tensor:
    return torch.sum((target-X)**2,dim=1)
