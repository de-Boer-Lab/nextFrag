import os
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=None)
def get_project_root() -> Path:
    """Get the project root directory.

    Checks the NEXTFRAG_ROOT environment variable first, then falls back to
    ~/.nextFrag/config.txt (written by: python -m nextFrag.setup /path/to/root).
    """
    if env_root := os.environ.get('NEXTFRAG_ROOT'):
        return Path(env_root)
    config_file = Path.home() / '.nextFrag' / 'config.txt'
    if not config_file.exists():
        raise RuntimeError(
            "nextFrag not configured.\n"
            "Run: python -m nextFrag.setup /path/to/root\n"
            "Or set the NEXTFRAG_ROOT environment variable."
        )
    return Path(config_file.read_text().strip())

DATASET_CONFIG = {
    'yeast': {'seqsize': 150, 'in_channels': 6, 'batch_sz': 256},
    'human': {'seqsize': 200, 'in_channels': 5, 'batch_sz': 32},
}

ARCH_CONFIG = {
    'rnn':  {'lr': 0.005, 'num_epochs': 80},
    'cnn':  {'lr': 0.005, 'num_epochs': 80},
    'attn': {'lr': 0.001, 'num_epochs': 80},
}