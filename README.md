# DNA Active Learning

A framework for benchmarking active learning strategies on DNA sequence-to-expression models.

## Installation
---
```bash
git clone https://github.com/de-Boer-Lab/dna_active_learning.git
cd dna_active_learning
pip install -e .
```

Set your project root directory:
```bash
python -m dna_active_learning.setup /path/to/your/data/root
```

## Repository Overview
---
**`models/`** - Implementation of DREAM Challenge model architectures (CNN, RNN, Attention) with training and evaluation utilities.

**`sequence_selection/`** - Active learning strategies including MC Dropout, k-means clustering, LCMD, and ensemble methods. Contains the main AL loop implementation.

**`data/`** - Demo datasets.

### File Structure

The package expects this directory structure:
```
{PROJECT_ROOT}/
└── {dataset}/
    ├── round_0/common/
    │   ├── train.txt
    │   └── pool.txt
    ├── round_{i}/{strategy}/{arch}_{seed}/
    │   ├── data/
    │   │   ├── selected.txt
    │   │   ├── train.txt
    │   │   └── pool.txt
    │   └── model/
    │       └── model_best.pth
    └── val.txt
```

## Basic Usage
---
### Run Active Learning Loop
```bash
python -m dna_active_learning.sequence_selection.al_loop \
    <dataset> <strategy> <arch> <seed> [OPTIONS]
```

**Example:**
```bash
# Run 3-round AL experiment with MC Dropout
python -m dna_active_learning.sequence_selection.al_loop \
    yeast mcd cnn 42 --num-rounds 3 --num-selected 20000
```

**Arguments:**
- `dataset`: Dataset name (`yeast` or `human`)
- `strategy`: AL strategy (`mcd`, `kmeans`, or `lcmd`)
- `arch`: Model architecture (`cnn`, `rnn`, or `attn`)
- `seed`: Random seed for reproducibility
- `--num-rounds`: Number of AL rounds (default: 3)
- `--num-selected`: Sequences to select per round (default: 20,000)
- `--start-round`: Resume from specific round (default: 1)

### Train a Single Model
```bash
# Using AL experiment structure
python -m dna_active_learning.models.train_model al <dataset> <arch> \
    --strategy <al_strategy> --round <round_num> --seed <seed>

# Using custom data paths
python -m dna_active_learning.models.train_model custom <dataset> <arch> \
    --train <train_path> --val <val_path> --model-dir <output_dir>
```

**Examples:**
```bash
# Train within AL structure
python -m dna_active_learning.models.train_model al yeast cnn \
    --strategy random --round 1 --seed 42

# Train with custom paths
python -m dna_active_learning.models.train_model custom human attn \
    --train data/my_train.txt --val data/my_val.txt --model-dir outputs/
```

### Select Sequences with Different Strategies

**Ensemble** - Disagreement-based selection across multiple models.
```bash
# Multi-architecture ensemble
python -m dna_active_learning.sequence_selection.ensemble multi \
    yeast all_arch --round 2 --seed 42

# Same-architecture ensemble
python -m dna_active_learning.sequence_selection.ensemble same \
    yeast cnn --round 2 --seeds 1 2 3 4 5
```

**MC Dropout (mcd)** - Uncertainty-based selection using Monte Carlo dropout to identify sequences where the model is most uncertain.
```bash
python -m dna_active_learning.sequence_selection.mc_dropout \
    yeast cnn 2 42 --num_passes 50 --num_selected 20000
```

**K-means (kmeans)** - Diversity-based selection using k-means clustering in embedding space to choose representative sequences.
```bash
python -m dna_active_learning.sequence_selection.diversity_strategies \
    yeast cnn kmeans 2 42 --num_selected 20000
```

**LCMD (lcmd)** - Iteratively selects cluster centers by identifying the largest cluster and choosing its furthest point, prioritizing maximally different sequences.
```bash
python -m dna_active_learning.sequence_selection.diversity_strategies \
    yeast cnn lcmd 2 42 --num_selected 20000
```