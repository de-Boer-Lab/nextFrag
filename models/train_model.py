import torch
import argparse
from pathlib import Path
from sequence_selection.dataloader import prepare_dataloader
from sequence_selection.evaluation import eval_model
from .trainer import Trainer
from .model_utils import init_model
from nextFrag.config import get_project_root, DATASET_CONFIG, ARCH_CONFIG

def train_al_model(
    dataset: str, 
    arch: str, 
    al_strategy: str,
    round_num: int,
    seed: int,
    **kwargs
):
    """
    Train a model as part of an active learning experiment.
    
    Uses the standard AL directory structure and handles path construction automatically.
    
    Args:
        dataset: Dataset name ('yeast' or 'human')
        arch: Model architecture ('cnn', 'rnn', or 'attn')
        al_strategy: Active learning strategy name
        round_num: AL round number
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to train_model()
    """
    project_root = get_project_root()
    experiment_path = project_root / dataset / f'round_{round_num}' / al_strategy / f'{arch}_{seed}'
    train_path = experiment_path / 'data' / 'train.txt'
    val_path = project_root / dataset / 'val.txt'
    model_dir = experiment_path / 'model'
    results_path = model_dir / 'results.txt'
    return train_model(
        dataset=dataset,
        arch=arch,
        train_path=train_path,
        val_path=val_path,
        model_dir=model_dir,
        results_path=results_path,
        seed=seed,
        **kwargs
    )

def train_model(
    dataset: str, 
    arch: str, 
    train_path: str | Path,
    val_path: str | Path,
    model_dir: str | Path,
    results_path: str | Path = None,
    seed: int = 42,
    num_epochs: int = 80,
    train_batch_sz: int = None,
    val_batch_sz: int = None,
    lr: float = None
):
    """
    Train a model with explicit data paths.
    
    For standard AL experiments, use train_al_model() instead.
    
    Args:
        dataset: Dataset name ('yeast' or 'human') - used for defaults
        arch: Model architecture ('cnn', 'rnn', or 'attn')
        train_path: Path to training data file
        val_path: Path to validation data file
        model_dir: Directory to save model checkpoints
        results_path: Path to save evaluation results (defaults to model_dir/results.txt)
        seed: Random seed for reproducibility
        num_epochs: Number of training epochs
        train_batch_sz: Training batch size (uses dataset default if None)
        val_batch_sz: Validation batch size (default: 4096)
        lr: Learning rate (uses architecture default if None)
    """
    model_dir=Path(model_dir)
    seqsize = DATASET_CONFIG[dataset]['seqsize']
    if results_path is None:
        results_path = model_dir / 'results.txt'
    if train_batch_sz is None:
        train_batch_sz = DATASET_CONFIG[dataset]['batch_sz']
    if val_batch_sz is None:
        val_batch_sz = 4096
    if lr is None:
        lr = ARCH_CONFIG[arch]['lr']
    if num_epochs is None:
        lr = ARCH_CONFIG[arch]['num_epochs']

    generator = torch.Generator()
    generator.manual_seed(seed)

    model=init_model(dataset=dataset,arch=arch)

    train_dl = prepare_dataloader(
        train_path, 
        seqsize=seqsize,
        dset=dataset,
        batch_size=train_batch_sz,
        shuffle=True,
        generator=generator
    )
    val_dl = prepare_dataloader(
        val_path, 
        seqsize=seqsize, 
        dset=dataset,
        batch_size=val_batch_sz,
        shuffle=False
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        model_dir=model_dir,
        num_epochs=num_epochs,
        lr=lr,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    trainer.fit()

    return eval_model(model_path=model_dir / 'model_best.pth',
                      out_file=results_path,
                      dataset=dataset,
                      arch=arch)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # ===== AL Mode =====
    al_parser = subparsers.add_parser('al',help='Train using AL experiment structure')
    al_parser.add_argument('dataset',choices=['yeast', 'human'])
    al_parser.add_argument('arch',choices=['rnn', 'cnn', 'attn'])
    al_parser.add_argument('--strategy',type=str,required=True)
    al_parser.add_argument('--round',type=int,required=True)
    al_parser.add_argument('--seed',type=int,required=True)
    al_parser.add_argument('--epochs',type=int)
    al_parser.add_argument('--lr',type=float)
    al_parser.add_argument('--train-batch-size',type=int)
    al_parser.add_argument('--val-batch-size',type=int)
    
    # ===== Custom Mode =====
    custom_parser = subparsers.add_parser('custom',help='Train with custom data paths')
    custom_parser.add_argument('dataset',choices=['yeast', 'human'])
    custom_parser.add_argument('arch',choices=['cnn', 'rnn', 'attn'])
    custom_parser.add_argument('--train',type=str,required=True)
    custom_parser.add_argument('--val',type=str,required=True)
    custom_parser.add_argument('--model-dir',type=str,required=True)
    custom_parser.add_argument('--results',type=str)
    custom_parser.add_argument('--seed',type=int,default=42)
    custom_parser.add_argument('--epochs',type=int,default=80)
    custom_parser.add_argument('--lr',type=float)
    custom_parser.add_argument('--train-batch-size',type=int)
    custom_parser.add_argument('--val-batch-size',type=int)
    
    args = parser.parse_args()
    
    if args.mode == 'al':
        print(f"Training AL model: {args.dataset}/{args.arch}")
        print(f"  Strategy: {args.strategy}, Round: {args.round}, Seed: {args.seed}")
        
        return train_al_model(
            dataset=args.dataset,
            arch=args.arch,
            al_strategy=args.strategy,
            round_num=args.round,
            seed=args.seed,
            num_epochs=args.epochs,
            lr=args.lr,
            train_batch_sz=args.train_batch_size,
            val_batch_sz=args.val_batch_size
        )
    else:
        print(f"Training custom model: {args.dataset}/{args.arch}")
        print(f"  Train: {args.train}")
        print(f"  Val: {args.val}")
        print(f"  Model dir: {args.model_dir}")
        
        return train_model(
            dataset=args.dataset,
            arch=args.arch,
            train_path=args.train,
            val_path=args.val,
            model_dir=args.model_dir,
            results_path=args.results,
            seed=args.seed,
            num_epochs=args.epochs,
            lr=args.lr,
            train_batch_sz=args.train_batch_size,
            val_batch_sz=args.val_batch_size
        )

if __name__ == "__main__":
    main()
