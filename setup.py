from pathlib import Path
import sys

def setup(root_dir=None):
    """Set up the project root directory."""
    if root_dir is None:
        root_dir = Path.cwd()
    else:
        root_dir = Path(root_dir).resolve()
    
    # Validate path exists
    if not root_dir.exists():
        print(f"Creating directory: {root_dir}")
        root_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config directory and file
    config_dir = Path.home() / '.nextFrag'
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / 'config.txt'
    with open(config_file, 'w') as f:
        f.write(str(root_dir))
    
    print(f"Project root set to: {root_dir}")
    print(f"Config saved to: {config_file}")

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else None
    setup(root)