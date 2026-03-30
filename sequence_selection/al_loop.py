"""
Active Learning Loop
====================
Main training loop for non-ensemble AL strategies (MC Dropout, k-means, LCMD).

The loop consists of three steps per round:
1. Select sequences using the AL strategy
2. Update training and pool datasets
3. Train a new model on the updated data
"""

import argparse
from models.train_model import train_al_model
from .utils import update_train_and_pool
from .mc_dropout import mc_dropout
from .diversity_strategies import diversity_al

def run_al_loop(
    dataset: str,
    strategy: str,
    arch: str,
    seed: int,
    num_rounds: int = 3,
    num_selected: int = 20_000,
    start_round: int = 1
):
    """
    Run the active learning loop for a given strategy.
    
    Args:
        dataset: Dataset name ('yeast' or 'human')
        strategy: AL strategy name ('mcd', 'kmeans', or 'lcmd')
        arch: Model architecture ('cnn', 'rnn', or 'attn')
        seed: Random seed for reproducibility
        num_rounds: Total number of AL rounds to run
        num_selected: Number of sequences to select per round
        start_round: Round number to start from
    """

    if strategy == 'mcd':
        selection_fn = mc_dropout
    else:  # kmeans or lcmd
        selection_fn = diversity_al

    if num_selected == 20_000:
        strategy_folder = strategy
    else:
        strategy_folder = f"{strategy}_{num_selected // 1000}k"

    print("=" * 40)
    print(f"Active Learning Loop - {strategy.upper()}")
    print("=" * 40)
    print(f"Dataset:       {dataset}")
    print(f"Architecture:  {arch}")
    print(f"Strategy:      {strategy}")
    print(f"Seed:          {seed}")
    print(f"Rounds:        {start_round} to {num_rounds}")
    print(f"Selected/round: {num_selected:,}")
    print("=" * 40)

    for round_num in range(start_round, num_rounds + 1):
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"\n[1/3] Selecting sequences using {strategy}...")
        selection_fn(
            dataset=dataset,
            arch=arch,
            strategy=strategy,
            round_num=round_num,
            seed=seed,
            num_selected=num_selected
        )
        print(f"Selected {num_selected:,} sequences")
        print(f"\n[2/3] Updating train and pool datasets...")
        update_train_and_pool(
            dataset=dataset,
            current_round=round_num,
            al_strategy=strategy_folder,
            arch=arch,
            seed=seed,
            num_rounds=num_rounds
        )
        print("Datasets updated")
        print(f"\n[3/3] Training model...")
        train_al_model(
            dataset=dataset,
            arch=arch,
            al_strategy=strategy_folder,
            round_num=round_num,
            seed=seed
        )
        print(f"Model trained")
        print(f"✓ Round {round_num} complete!")
        print(f"{'='*40}")
    
    print(f"\n{'='*40}")
    print("Active Learning Loop Complete!")
    print(f"{'='*40}")
    print(f"Completed {num_rounds - start_round + 1} rounds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",choices=['yeast','human'])
    parser.add_argument("strategy", choices=['mcd','kmeans','lcmd'])
    parser.add_argument("arch",choices=['rnn','cnn','attn'])
    parser.add_argument("seed",type=int)
    parser.add_argument("--num-rounds","-r",type=int,default=3)
    parser.add_argument("--num-selected", "-n", type=int, default=20_000)
    parser.add_argument("--start-round","-s",type=int,default=1)
    args = parser.parse_args()

    if args.start_round > args.num_rounds:
        parser.error(f"--start-round ({args.start_round}) cannot be greater than --num-rounds ({args.num_rounds})")
    if args.start_round < 1:
        parser.error(f"--start-round must be >= 1")
    if args.num_selected <= 0:
        parser.error(f"--num-selected must be positive")

    run_al_loop(
        dataset=args.dataset,
        strategy=args.strategy,
        arch=args.arch,
        seed=args.seed,
        num_rounds=args.num_rounds,
        num_selected=args.num_selected,
        start_round=args.start_round
    )

if __name__=="__main__":
    main()
