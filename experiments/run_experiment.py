import argparse
import yaml
from src.training import train_model

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    train_model(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)