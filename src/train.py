import os
import hydra
import logging
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter  # Import TensorBoardX

from data.dataset import TBIDataset
from data.preprocessing import DataPreprocessor
import models
from utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="default")
def train(config: DictConfig):
    # Set random seed
    torch.manual_seed(config.seed)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
    
    # Load and preprocess data
    data = pd.read_csv(config.data.path)
    preprocessor = DataPreprocessor(config)
    
    # Split data
    train_data, val_data = train_test_split(
        data, 
        test_size=config.validation.split_ratio,
        random_state=config.seed
    )
    
    # Preprocess data
    preprocessor.fit(train_data)
    train_processed = preprocessor.transform(train_data)
    val_processed = preprocessor.transform(val_data)
    
    # Create datasets
    train_dataset = TBIDataset(train_processed, config)
    val_dataset = TBIDataset(val_processed, config)
    
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
    model = hydra.utils.instantiate(config.model)
    model = model.to(config.device)
    
    # Get optimizer and scheduler
    optim_config = model.configure_optimizers(config)
    optimizer = optim_config['optimizer']
    scheduler = optim_config['scheduler']
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % config.logging.log_interval == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                          f'Loss: {loss.item():.6f}')
                # Log training loss to TensorBoard
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config.device), target.to(config.device)
                output = model(data)
                val_loss += torch.nn.functional.cross_entropy(output, target).item()
                
                val_predictions.extend(output.argmax(dim=1).cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        metrics = calculate_metrics(val_targets, val_predictions)
        
        # Log metrics
        logger.info(f'Epoch {epoch}: Val Loss: {val_loss:.4f}, Metrics: {metrics}')
        # Log validation loss and metrics to TensorBoard
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f'Validation/{metric_name}', metric_value, epoch)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model
            if epoch % config.logging.save_interval == 0:
                model.save(os.path.join(config.output_dir, 'models', f'model_epoch_{epoch}.pt'))
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping.patience:
                logger.info(f'Early stopping triggered after {epoch} epochs')
                break
        
        # Update scheduler
        scheduler.step(val_loss)
    
    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    train() 