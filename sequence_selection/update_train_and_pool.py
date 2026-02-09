from pathlib import Path
from ..config import PROJECT_ROOT

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
    dataset_root = PROJECT_ROOT / dataset
    
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
        