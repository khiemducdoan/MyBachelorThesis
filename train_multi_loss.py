import os
import hydra
import logging
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import random

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
import wandb
from src.utils.metrics import set_naim_params
from src.data.dataset import TBIDataset
from src.utils.metrics import calculate_metrics


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

categorical_features =  ['sex', 'tbi_cli_reason', 'tbi_cli_awaken', 'tbi_cli_headache', 'tbi_cli_blue', 'tbi_cli_para_ner', 'tbi_cli_quadriplegia', 
 'tbi_cli_epileptic', 'tbi_cli_stiff_neck', 'tbi_cli_dam_chest_abdomen', 'tbi_cli_recall', 'tbi_cli_pupils_left_reflex', 
 'tbi_cli_pupils_right_reflex', 'tbi_cli_diabetes', 'tbi_cli_hypertension', 'tbi_cli_stroke', 'tbi_cli_cardiovascular', 
 'tbi_ct_brain_parenchyma___1', 'tbi_ct_brain_parenchyma___2', 'tbi_ct_brain_parenchyma___3', 'tbi_ct_brain_parenchyma___4', 
 'tbi_ct_brain_parenchyma___5', 'tbi_ct_brain_parenchyma___6', 'tbi_ct_brain_parenchyma___8', 'tbi_ct_epidural_hematoma_proportion', 
 'tbi_ct_subdural_hematoma_position_proprotion', 'tbi_ct_blood_hematoma_proportion', 'tbi_ct_subarachnoid_characteristic', 
 'tbi_ct_bottom_tank_characteristic', 'tbi_ct_skull_fracture_characteristic', 'tbi_ct_skull_risk']
logger = logging.getLogger(__name__)
wandb.login()
# @hydra.main(config_path="./configs", config_name="default")
def train(config):
    
    wandb.init(
        project=config.logging.wandb.project)
    # ========================================================Set random seed========================================================
    print("Training started...")
    torch.manual_seed(config.seed)
    
    # ========================================================Initialize TensorBoard writer========================================================
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
    
    # ========================================================Load and preprocess data========================================================
    data = pd.read_csv(config.data.data_path)

    # ========================================================Split data======================================================
    train_data, val_data = train_test_split(
        data, 
        test_size=config.validation.split_ratio,
        random_state=config.seed,
        stratify=data[config.data.caller.target_column]
    )

    # ==========================================================Create datasets========================================
    if config.data.name == "tbi":
        config.data.caller.categorical_features = categorical_features if config.data.name == "tbi" else None
    train_dataset = hydra.utils.instantiate(config.data.caller, data = train_data)
    val_dataset = hydra.utils.instantiate(config.data.caller, data = val_data)
    # =====================================================Create dataloaders================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size
    )
    
    # =======================================================Initialize model============================================
    model = hydra.utils.instantiate(config.model.model)
    torch.cuda.empty_cache()
    model = model.to(config.device)
    
    # Get optimizer and scheduler
    optim_config = model.configure_optimizers(config.model)
    optimizer = optim_config['optimizer']
    scheduler = optim_config['scheduler']
    # =============================================UNCOMMENT IF YOU WANT TO USE CLASS WEIGHTS=====================================================================
    # Compute class weights
    # classes = np.unique(train_data[config.data.caller.target_column])
    # class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_data[config.data.caller.target_column])
    # config.model.loss.weight = class_weights.tolist()
    loss_fn = model.configure_loss(config.model)
    # ====================================================Training loop=====================================================================
    best_val_loss = float('inf')
    best_accuracy = 0.0
    patience_counter = 0
    best_conf_matrix = None  # Lưu confusion matrix tốt nhất
    # =========================================================================================================================
    for epoch in range(config.training.num_epochs):
        # Training
        model.train()
        train_loss = {"loss": 0, "loss_naim": 0, "loss_vitbi": 0, "loss_combined": 0}
        train_predictions = []
        train_targets = []
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch_idx, batch in enumerate(train_bar):
            features,target = batch
            target = target.to(config.device)
            optimizer.zero_grad()
            if isinstance(features, list):
                features = [f.to(config.device) for f in features]
                output1, output2, output3 = model(*features)
            else:
                features = features.to(config.device)
                output1, output2, output3 = model(features)
            
            loss = loss_fn(output1, output2, output3, target)
            loss["loss"].backward()
            optimizer.step()
            
            # Track individual losses
            train_loss["loss"] += loss["loss"].item()
            train_loss["loss_naim"] += loss["loss_naim"].item()
            train_loss["loss_vitbi"] += loss["loss_vitbi"].item()
            train_loss["loss_combined"] += loss["loss_combined"].item()
            
            # Track predictions for accuracy calculation
            train_predictions.extend(output3.argmax(dim=1).cpu().numpy())  # Using combined output for predictions
            train_targets.extend(target.cpu().numpy())
            train_accuracy = calculate_metrics(train_targets, train_predictions)
            if batch_idx % config.logging.log_interval == 0:
                # logger.info(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                #           f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                #           f'Loss total loss: {loss["loss"].item():.6f}')
                # writer.add_scalar('Train/Loss Total', loss["loss"].item(), epoch * len(train_loader) + batch_idx)
                wandb.log({"Train Loss Total": loss["loss"].item()})  # Log to wandb
        #wandb log training accuracy 

        # ====================================================Validation====================================================
        model.eval()
        val_loss = {"loss": 0, "loss_naim": 0, "loss_vitbi": 0, "loss_combined": 0}
        val_predictions1 = []  # NAIM predictions
        val_predictions2 = []  # ViTBI predictions
        val_predictions3 = []  # Combined predictions
        val_targets = []
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                features, target = batch
                target = target.to(config.device)
                if isinstance(features, list):
                    features = list(f.to(config.device) for f in features)
                    output1, output2, output3 = model(*features)
                else:
                    features = features.to(config.device)
                    output1, output2, output3 = model(features)
                
                loss = loss_fn(output1, output2, output3, target)
                
                # Track losses
                val_loss["loss"] += loss["loss"].item()
                val_loss["loss_naim"] += loss["loss_naim"].item()
                val_loss["loss_vitbi"] += loss["loss_vitbi"].item()
                val_loss["loss_combined"] += loss["loss_combined"].item()
                
                # Track predictions for all outputs
                val_predictions1.extend(output1.argmax(dim=1).cpu().numpy())
                if output2 is not None:  # Only if text data is present
                    val_predictions2.extend(output2.argmax(dim=1).cpu().numpy())
                val_predictions3.extend(output3.argmax(dim=1).cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        # Calculate average losses
        num_batches = len(val_loader)
        val_loss = {k: v / num_batches for k, v in val_loss.items()}
        
        # Calculate metrics for all outputs
        metrics_naim = calculate_metrics(val_targets, val_predictions1)
        metrics_vitbi = calculate_metrics(val_targets, val_predictions2) if len(val_predictions2) > 0 else None
        metrics_combined = calculate_metrics(val_targets, val_predictions3)
        
        # Log validation metrics
        # logger.info(f'Epoch {epoch}:')
        # logger.info(f'Total Val Loss: {val_loss["loss"]:.4f}')
        # logger.info(f'NAIM - Loss: {val_loss["loss_naim"]:.4f}, Metrics: {metrics_naim}')
        # if metrics_vitbi:
        #     logger.info(f'ViTBI - Loss: {val_loss["loss_vitbi"]:.4f}, Metrics: {metrics_vitbi}')
        # logger.info(f'Combined - Loss: {val_loss["loss_combined"]:.4f}, Metrics: {metrics_combined}')
        
        # Log to wandb
        wandb_logs = {
            "Validation Loss Total": val_loss["loss"],
            "Validation Loss NAIM": val_loss["loss_naim"],
            "Validation Loss Combined": val_loss["loss_combined"],
            
            # NAIM metrics
            "Validation NAIM Accuracy": metrics_naim["accuracy"],
            "Validation NAIM F1 ": metrics_naim["f1_weighted"],
            "Validation NAIM Precision": metrics_naim["precision_macro"],
            "Validation NAIM Recall": metrics_naim["recall_macro"],
            
            # Combined metrics
            "Validation Combined Accuracy": metrics_combined["accuracy"],
            "Validation Combined F1 ": metrics_combined["f1_weighted"],
            "Validation Combined Precision": metrics_combined["precision_macro"],
            "Validation Combined Recall": metrics_combined["recall_macro"],
        }
        
        # Add ViTBI metrics if available
        if metrics_vitbi:
            wandb_logs.update({
                "Validation Loss ViTBI": val_loss["loss_vitbi"],
                "Validation ViTBI Accuracy": metrics_vitbi["accuracy"],
                "Validation ViTBI F1": metrics_vitbi["f1_weighted"],
                "Validation ViTBI Precision": metrics_vitbi["precision_macro"],
                "Validation ViTBI Recall": metrics_vitbi["recall_macro"],
            })
        # Log to TensorBoard
        # writer.add_scalar('Validation/Loss Total', val_loss["loss"], epoch)
        # writer.add_scalar('Validation/Loss NAIM', val_loss["loss_naim"], epoch)
        # writer.add_scalar('Validation/Loss Combined', val_loss["loss_combined"], epoch)
        # if metrics_vitbi:
        #     writer.add_scalar('Validation/Loss ViTBI', val_loss["loss_vitbi"], epoch)
        wandb.log(wandb_logs)
        # Use combined metrics for best model selection and early stopping
        val_accuracy = metrics_combined["accuracy"]
        wandb.log({
            "Accuracy/Train": train_accuracy,
            "Accuracy/Test": val_accuracy,
        }, step=epoch)
        # ====================================================Log best accuracy to wandb====================================================
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            wandb.log({"Best Accuracy": best_accuracy})  # Log best accuracy to wandb
            best_conf_matrix = confusion_matrix(val_targets, val_predictions3)
            
            # Vẽ confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix (Epoch {epoch})')
            # torch.save(model, os.path.join(config.output_dir, f'model_Best_{epoch}.pt'))
            # logger.info(f'Saved best model at epoch {epoch} with accuracy: {best_accuracy:.4f}')
            # Ghi confusion matrix vào TensorBoard
            # Lưu hình ảnh confusion matrix
            cm_path = os.path.join(config.output_dir, f'best_confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()

            # Ghi confusion matrix vào TensorBoard
            # writer.add_image('Validation/Confusion Matrix', 
            #                  torch.tensor(plt.imread(cm_path)), epoch, dataformats='HWC')
            wandb.log({"Confusion Matrix": wandb.Image(cm_path)})  # Log confusion matrix to wandb

        # Early stopping
        # if val_loss["loss"] < best_val_loss:
        #     best_val_loss = val_loss["loss"]
        #     patience_counter = 0
        #     if epoch % config.logging.save_interval == 0:
        #         # model.save(os.path.join(config.output_dir, f'model_{epoch}.pt'))
        #         logger.info(f'Saved model at epoch {epoch}')
        # else:
        #     patience_counter += 1
        #     if patience_counter >= config.training.early_stopping.patience:
        #         logger.info(f'Early stopping triggered after {epoch} epochs')
        #         break
        
        # # Update scheduler
        # scheduler.step(val_loss)
    
    # ====================================================Lưu confusion matrix tốt nhất====================================================
    if best_conf_matrix is not None:
        # logger.info(f'Saving best confusion matrix with accuracy: {best_accuracy:.4f}')
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
                'values': [4, 8, 16, 32, 64, 128, 256]#ssible values for batch size
            },
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': -8,
                'max': -4
            },
            'd_token': {
                'values': [8, 16, 32, 64]  # Possible values for d_token
            },
            "Fdropout_rate": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.5
            },
            "Mdropout_rate": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.4
            },
            'Fnum_heads': {
                'values': [1, 2, 4, 8]  # Possible values for number of heads
            },
            # 'Mnum_heads': {
            #     'values': [1, 2, 4, 8]  # Possible values for number of heads
            # },
            # "num_layers": {
            #     "values": [1, 2, 3,4,5,6,7,8,9,10]  # Possible values for number of layers
            # },
            'Fnum_layers': {
                'values': [1, 2, 3, 4, 5, 6, 7, 8]  # Possible values for number of layers
            },
            'Mnum_layers': {
                'values': [1, 2, 3, 4]  # Possible values for number of layers
            },
            'pretrained_model_name': {
                'values': ["emilyalsentzer/Bio_ClinicalBERT",
                           "dmis-lab/biobert-base-cased-v1.1",
                           "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
                           "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                           ]  # Possible values for pretrained model
            }
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
        #================feature transformer=========================
        config.model.model.params_naim.num_layers = sweep_params.Fnum_layers
        config.model.model.params_naim.d_token = sweep_params.d_token
        config.model.model.params_naim.dropout_rate = sweep_params.Fdropout_rate
        config.model.model.params_naim.num_heads = sweep_params.Fnum_heads
        #================maske transformer=========================
        # config.model.model.params.num_layers = sweep_params.Fnum_layers
        # config.model.model.mask_params.d_token = sweep_params.d_token
        # config.model.model.mask_params.dropout_rate = sweep_params.Fdropout_rate
        # config.model.model.mask_params.num_heads = sweep_params.Fnum_heads
        #================text branch=========================
        config.model.model.params_vibert.dropout_rate = sweep_params.Mdropout_rate
        config.model.model.params_vibert.num_layers = sweep_params.Mnum_layers
        # config.model.model.params_vibert.pretrained_model_name = sweep_params.pretrained_model_name
        # config.data.caller.tokenizer = sweep_params.pretrained_model_name
        #Call the train function with updated config
        train(config)

    # Start the sweep agent
    wandb.agent(sweep_id, function=sweep_train,count=config.logging.sweep_count)


@hydra.main(config_path="./configs", config_name="main")
def main(config):
    HYDRA_FULL_ERROR=1
    if not config.logging.sweep:
        
        train(config.default)
    else:
        train_with_sweep(config.default_sweep)
if __name__ == "__main__":
    main()
