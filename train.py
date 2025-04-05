import os
import hydra
import logging
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
import wandb
from src.data.dataset import TBIDataset
from src.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)
wandb.login()
# @hydra.main(config_path="./configs", config_name="default")
def train(config):
    # Set random seed
    print("Training started...")
    torch.manual_seed(config.seed)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
    
    # Load and preprocess data
    data = pd.read_csv(config.data.data_path)

    # Split data
    train_data, val_data = train_test_split(
        data, 
        test_size=config.validation.split_ratio,
        random_state=config.seed,
        stratify=data[config.data.caller.target_column]
    )

    # Create datasets
    train_dataset = hydra.utils.instantiate(config.data.caller, data = train_data)
    val_dataset = hydra.utils.instantiate(config.data.caller, data = val_data)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size
    )
    
    # Initialize model
    model = hydra.utils.instantiate(config.model.model)
    model = model.to(config.device)
    
    # Get optimizer and scheduler
    optim_config = model.configure_optimizers(config.model)
    optimizer = optim_config['optimizer']
    scheduler = optim_config['scheduler']
    
    # Training loop
    best_val_loss = float('inf')
    best_accuracy = 0.0
    patience_counter = 0
    best_conf_matrix = None  # Lưu confusion matrix tốt nhất

    for epoch in range(config.training.num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch_idx, batch in enumerate(train_bar):
            features,target = batch
            target = target.to(config.device)
            optimizer.zero_grad()
            if isinstance(features, list):
            # Assume model expects multiple inputs
                features = [f.to(config.device) for f in features]
                output = model(*features)
            else:
            # Assume model expects a single input
                features = features.to(config.device)
                output = model(features)
            
            # lets go
            # optimizer.zero_grad()
            # output = model(*features)  # Pass all feature sets to the model
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % config.logging.log_interval == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                          f'Loss: {loss.item():.6f}')
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                features,target = batch
                target = target.to(config.device)
                if isinstance(features, list):
            # Assume model expects multiple inputs
                    features = list(f.to(config.device) for f in features)
                    output = model(*features)
                else:
            # Assume model expects a single input
                    features = features.to(config.device)
                    output = model(features)
                
                # *features, target = [item.to(config.device) for item in batch]
                # output = model(*features)  # Pass all feature sets to the model
                val_loss += torch.nn.functional.cross_entropy(output, target).item()
                
                val_predictions.extend(output.argmax(dim=1).cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        metrics = calculate_metrics(val_targets, val_predictions)
        val_accuracy = metrics.get("accuracy", 0.0)
        # Log validation loss and metrics
        logger.info(f'Epoch {epoch}: Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Metrics: {metrics}')
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f'Validation/{metric_name}', metric_value, epoch)
        
        # Log best accuracy to wandb
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            wandb.log({"Best Accuracy": best_accuracy})  # Log best accuracy to wandb
            best_conf_matrix = confusion_matrix(val_targets, val_predictions)
            
            # Vẽ confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix (Epoch {epoch})')

            # Lưu hình ảnh confusion matrix
            cm_path = os.path.join(config.output_dir, f'best_confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()

            # Ghi confusion matrix vào TensorBoard
            writer.add_image('Validation/Confusion Matrix', 
                             torch.tensor(plt.imread(cm_path)), epoch, dataformats='HWC')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if epoch % config.logging.save_interval == 0:
                model.save(os.path.join(config.output_dir, f'model_epoch_{epoch}.pt'))
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping.patience:
                logger.info(f'Early stopping triggered after {epoch} epochs')
                break
        
        # Update scheduler
        scheduler.step(val_loss)
    
    # Lưu confusion matrix tốt nhất
    if best_conf_matrix is not None:
        logger.info(f'Saving best confusion matrix with accuracy: {best_accuracy:.4f}')
        cm_path_final = os.path.join(config.output_dir, 'final_best_confusion_matrix.png')
        plt.figure(figsize=(8, 6))
        sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Best Confusion Matrix (Accuracy: {best_accuracy:.4f})')
        plt.savefig(cm_path_final)
        plt.close()
    
    # Close TensorBoard writer
    writer.close()


def train_with_sweep(config):
    # Define sweep configuration
    sweep_config = {
        "name": config.logging.wandb.name,
        'method': 'bayes',  # Optimization method (e.g., grid, random, bayes)
        'metric': {
            'name': 'Best Accuracy',  # Metric to optimize
            'goal': 'maximize'  # Goal: maximize or minimize
        },
        'parameters': {
            'batch_size': {
                'values': [16, 32, 64]  # Possible values for batch size
            },
            'learning_rate': {
                'min': 0.0001,  # Minimum value for learning rate
                'max': 0.1   # Maximum value for learning rate
            },
            "num_layers": {
                "values": [1, 2, 3,4,5,6,7,8,9,10]  # Possible values for number of layers
            },
            }
        }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=config.logging.wandb.project)

    def sweep_train():
        # Initialize wandb with sweep configuration
        wandb.init()
        sweep_params = wandb.config

        # Update config with sweep parameters
        config.training.batch_size = sweep_params.batch_size
        config.model.optimizer.lr = sweep_params.learning_rate
        config.model.model.params.num_layers = sweep_params.num_layers

        # Call the train function with updated config
        train(config)

    # Start the sweep agent
    wandb.agent(sweep_id, function=sweep_train,count=config.logging.sweep_count)


@hydra.main(config_path="./configs", config_name="main")
def main(config):
    if not config.logging.sweep:
        train(config.default)
    else:
        train_with_sweep(config.default_sweep)
if __name__ == "__main__":
    main()
