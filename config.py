from pathlib import Path

def get_project_root():
    """Get the project root directory from config file."""
    config_file = Path.home() / '.dna_active_learning' / 'config.txt'
    
    if not config_file.exists():
        raise RuntimeError(
            "DNA Active Learning not configured.\n"
            "Run: python -m dna_active_learning.setup /path/to/root"
        )
    
    with open(config_file, 'r') as f:
        return Path(f.read().strip())

PROJECT_ROOT = get_project_root()