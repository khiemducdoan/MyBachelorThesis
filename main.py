import os
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from src.train import train

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument('--model', type=str, default='tabnet', help='Model to use for training')
    parser.add_argument('--data', type=str, default='tbi', help='Dataset to use for training')
    parser.add_argument('--experiment', type=str, default='exp001', help='Experiment configuration to use')
    return parser.parse_args()

@hydra.main(config_path="./configs", config_name="default")
def main(config: DictConfig):
    args = parse_args()
    
    # Override configurations with command-line arguments
    config.model = args.model
    config.data = args.data
    config.experiment = args.experiment
    
    # Ensure the output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set the device
    if not torch.cuda.is_available() and config.device == 'cuda':
        print("CUDA is not available. Switching to CPU.")
        config.device = 'cpu'
    
    # Print the final configuration
    print(OmegaConf.to_yaml(config))
    
    # Train the model
    train(config)

if __name__ == "__main__":
    main() 