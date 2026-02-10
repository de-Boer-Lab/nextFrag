import pandas as pd
import argparse
from pathlib import Path
from models.model_utils import load_model
from models.dl_utils import prepare_dataloader
from .utils import IPCA, _kmeans, LCMD, write_selections
from ..config import PROJECT_ROOT

def diversity_al(
    dataset: str,
    arch: str,
    strategy: str,
    round_num: int,
    seed: int,
    num_selected: int=20_000,
    num_pca_components: int=64,
    batch_size: int=2048
):
    strategy=strategy.lower()
    if strategy != 'kmeans' and strategy != 'lcmd':
        raise ValueError(f"Only k-means and LCMD are supported. Received {strategy}.")

    data_path = PROJECT_ROOT / dataset / f'round_{round_num}' / strategy / f'{arch}_{seed}' / 'data' / 'pool.txt'
    seqsize = 200 if dataset == 'human' else 150

    dataloader = prepare_dataloader(
        data_path, 
        seqsize=seqsize,
        dset=dataset,
        batch_size=batch_size,
        shuffle = False,
        revcomp_same_batch = True
    )

    model=load_model(dataset=dataset,al_strategy=strategy,arch=arch,seed=seed,round_num=round_num-1)

    post_pca=IPCA(model=model,
                  dataloader=dataloader, 
                  n_components=num_pca_components,
                  batch_size=batch_size//2)
    
    if strategy == 'kmeans':
        selected_idx = _kmeans(post_pca, num_selected=num_selected)
    else: # lcmd
        selected_idx = LCMD(post_pca,num_clusters=num_selected)

    df=pd.read_csv(data_path,sep='\t',header=None)
    df=df.iloc[selected_idx]

    if num_selected == 20_000:
        folder_name = strategy
    else:
        n_selected=num_selected//1000
        folder_name = f"{strategy}_{n_selected}k"

    out_path = PROJECT_ROOT / dataset / f'round_{round_num}' / folder_name / f'{arch}_{seed}' / 'data' / 'selected.txt'
    write_selections(out_path,df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",choices=['yeast','human'])
    parser.add_argument("arch",choices=['cnn', 'rnn', 'attn'])
    parser.add_argument("strategy",type=str,choices=['kmeans','lcmd'])
    parser.add_argument("round",type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("--num_selected",type=int,default=20_000)
    parser.add_argument("--num_pca_components",type=int,default=64)
    parser.add_argument("--batch_size",type=int,default=4096)
    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"{name}: {value}")
        
    diversity_al(
        dataset=args.dataset,
        arch=args.arch,
        strategy=args.strategy,
        round_num=args.round,
        seed=args.seed,
        num_selected=args.num_selected,
        num_pca_components=args.num_pca_components,
        batch_size=args.batch_size
    )
    print("\nSelection complete!")

if __name__ == "__main__":
    main()