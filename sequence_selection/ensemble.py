import numpy as np
import pandas as pd
import torch, argparse, copy
from typing import List
from pathlib import Path
from .utils import write_selections
from models.model_utils import load_model
from models.dl_utils import prepare_dataloader
from ..config import PROJECT_ROOT

def ensemble_multi_arch(
    dataset: str,
    composition: str,
    round_num: int,
    seed: int,
    num_selected: int=20_000,
    batch_size: int=2048
):
    data_path = PROJECT_ROOT / dataset / f'round_{round_num}' / composition / f'rnn_{seed}' / 'data' / 'pool.txt'
    seqsize = 200 if dataset == 'human' else 150
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = prepare_dataloader(
        data_path, 
        seqsize=seqsize,
        dset=dataset,
        batch_size=batch_size,
        shuffle = False
    )
    models = {}
    
    if composition != 'cnn-attn':
        rnn = load_model(
            dataset=dataset,
            al_strategy=composition,
            arch='rnn',
            seed=seed,
            round_num=round_num - 1
        )
        models['rnn'] = rnn.to(device).eval()
    
    if composition != 'rnn-attn':
        cnn = load_model(
            dataset=dataset,
            al_strategy=composition,
            arch='cnn',
            seed=seed,
            round_num=round_num - 1
        )
        models['cnn'] = cnn.to(device).eval()
    
    if composition != 'rnn-cnn':
        attn = load_model(
            dataset=dataset,
            al_strategy=composition,
            arch='attn',
            seed=seed,
            round_num=round_num - 1
        )
        models['attn'] = attn.to(device).eval()

    df=pd.read_csv(data_path,header=None,sep='\t')
    num_seqs=len(df)

    all_var=[]
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)

            predictions = []
            if 'cnn' in models:
                predictions.append(models['cnn'].predict(X))
            if 'rnn' in models:
                predictions.append(models['rnn'].predict(X))
            if 'attn' in models:
                predictions.append(models['attn'].predict(X))

            if composition == 'all_arch':
                combined = torch.stack(predictions, dim=0)
                var = torch.var(combined, dim=0)
                var = var.cpu().numpy()
            else:
                var = torch.abs(predictions[0] - predictions[1]).cpu().numpy()

            all_var.append(var)

    all_var=np.concatenate(all_var)
    all_var=all_var.reshape(2,num_seqs)
    all_var=np.max(all_var,axis=0) # take max variance between sequence and reverse complement
    
    df['var']=all_var
    result_df=df.sort_values(by=['var'],ascending=False).head(num_selected)

    if num_selected == 20_000:
        folder_name = composition
    else:
        n_selected=num_selected//1000
        folder_name = f"{composition}_{n_selected}k"
    
    out_path = PROJECT_ROOT / dataset / f'round_{round_num}' / folder_name / f'rnn_{seed}' / 'data' / 'selected.txt'    
    write_selections(out_path,result_df)
    for arch in ['cnn','attn']:
        link = out_path.parent.parent / f"{arch}_{seed}"
        link.symlink_to(out_path.parent, target_is_directory=True)

def ensemble_same_arch(
    dataset: str,
    arch: str,
    round_num: int,
    seeds: List[int],
    num_selected: int=20_000,
    batch_size = 4096
):
    data_path = PROJECT_ROOT / dataset / f'round_{round_num}' / 'same_arch' / f'{arch}_{seeds[0]}' / 'data' / 'pool.txt'
    seqsize = 200 if dataset == 'human' else 150
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = prepare_dataloader(
        data_path, 
        seqsize=seqsize,
        species=dataset,
        batch_size=batch_size,
        shuffle = False
    )

    models=[]
    for seed in seeds:
        model=load_model(dataset=dataset, al_strategy='same_arch', arch=arch, seed=seed, round_num=round_num-1)
        models.append(copy.deepcopy(model))  

    for model in models:
        model = model.to(device).eval()

    df=pd.read_csv(data_path,header=None,sep='\t')
    num_seqs=len(df)

    all_var=[]
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)

            model_preds=[]
            for model in models:
                model_preds.append(model.predict(X))

            combined = torch.stack(model_preds).cpu().numpy()
            var = np.var(combined,axis=0)
            all_var.append(var)
            
    all_var=np.concatenate(all_var)
    all_var=all_var.reshape(2,num_seqs)
    all_var=np.max(all_var,axis=0) # take max variance between sequence and reverse complement

    df['var']=all_var
    df=df.sort_values(by=['var'],ascending=False)
    result_df=df[:num_selected]

    if num_selected == 20_000:
        folder_name = 'same_arch'
    else:
        n_selected=num_selected//1000
        folder_name = f"same_arch_{n_selected}k"

    out_path = PROJECT_ROOT / dataset / f'round_{round_num}' / folder_name / f'{arch}_{seeds[0]}' / 'data' / 'selected.txt'
    write_selections(out_path,result_df)
    for seed in seeds:
        if seed != seeds[0]:
            link = out_path.parent.parent / f"{arch}_{seed}"
            link.symlink_to(out_path.parent, target_is_directory=True)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # ===== Multi-Architecture Ensemble =====
    multi_parser = subparsers.add_parser('multi',help='Ensemble across different architectures')
    multi_parser.add_argument('dataset',choices=['yeast', 'human'])
    multi_parser.add_argument('composition',choices=['all_arch', 'rnn-cnn', 'rnn-attn', 'cnn-attn'])
    multi_parser.add_argument('--round',type=int,required=True)
    multi_parser.add_argument('--seed',type=int,required=True)
    multi_parser.add_argument('--num-selected',type=int,default=20_000)
    multi_parser.add_argument('--batch-size',type=int,default=2048)
    
    # ===== Same Architecture Ensemble =====
    same_parser = subparsers.add_parser('same',help='Ensemble across same architecture with different seeds')
    same_parser.add_argument('dataset',choices=['yeast', 'human'])
    same_parser.add_argument('arch',choices=['cnn', 'rnn', 'attn'])
    same_parser.add_argument('--round',type=int,required=True)
    same_parser.add_argument('--seeds',type=int,nargs='+',required=True)
    same_parser.add_argument('--num-selected',type=int,default=20_000)
    same_parser.add_argument('--batch-size',type=int,default=4096)

    args = parser.parse_args()
    
    if args.mode == 'multi':
        print(f"Multi-architecture ensemble: {args.composition}")
        print(f"  Dataset: {args.dataset}, Round: {args.round}, Seed: {args.seed}")
        
        ensemble_multi_arch(
            dataset=args.dataset,
            composition=args.composition,
            round_num=args.round,
            seed=args.seed,
            num_selected=args.num_selected,
            batch_size=args.batch_size
        )
    else:  # same
        print(f"Same-architecture ensemble: {args.arch}")
        print(f"  Dataset: {args.dataset}, Round: {args.round}")
        print(f"  Seeds: {args.seeds}")
        
        ensemble_same_arch(
            dataset=args.dataset,
            arch=args.arch,
            round_num=args.round,
            seeds=args.seeds,
            num_selected=args.num_selected,
            batch_size=args.batch_size
        )
    print("\nSelection complete!")

if __name__ == "__main__":
    main()