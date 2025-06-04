import os
import hydra
import logging
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
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
#write something
from src.utils.metrics import set_naim_params
from src.data.dataset import TBIDataset
from src.utils.metrics import calculate_metrics
categorical_features =  ['sex', 'tbi_cli_reason', 'tbi_cli_awaken', 'tbi_cli_headache', 'tbi_cli_blue', 'tbi_cli_para_ner', 'tbi_cli_quadriplegia', 
 'tbi_cli_epileptic', 'tbi_cli_stiff_neck', 'tbi_cli_dam_chest_abdomen', 'tbi_cli_recall', 'tbi_cli_pupils_left_reflex', 
 'tbi_cli_pupils_right_reflex', 'tbi_cli_diabetes', 'tbi_cli_hypertension', 'tbi_cli_stroke', 'tbi_cli_cardiovascular', 
 'tbi_ct_brain_parenchyma___1', 'tbi_ct_brain_parenchyma___2', 'tbi_ct_brain_parenchyma___3', 'tbi_ct_brain_parenchyma___4', 
 'tbi_ct_brain_parenchyma___5', 'tbi_ct_brain_parenchyma___6', 'tbi_ct_brain_parenchyma___8', 'tbi_ct_epidural_hematoma_proportion', 
 'tbi_ct_subdural_hematoma_position_proprotion', 'tbi_ct_blood_hematoma_proportion', 'tbi_ct_subarachnoid_characteristic', 
 'tbi_ct_bottom_tank_characteristic', 'tbi_ct_skull_fracture_characteristic', 'tbi_ct_skull_risk', 'hong_cau_v2', 'bach_cau_v2',
 'tieu_cau_v2', 'd_1_hst', 'ast_v2', 'alt_v2', 'd_2_protein', 'albumin_v2', 'ure_v2', 'creatinin_v2', 'prothrombin_v2', 
 'd_3_aptt', 'd_4_dtim']
logger = logging.getLogger(__name__)
@hydra.main(config_path="configs/default", config_name="default")
def train(config):
    try:
        data = pd.read_csv(config.data.data_path)
        print(config.data)
            # ========================================================Split data======================================================
        train_data, val_data = train_test_split(
            data, 
            test_size=config.validation.split_ratio,
            random_state=config.seed,
            stratify=data[config.data.caller.target_column]
        )
        # ==========================================================Create datasets========================================
        train_dataset = TBIDataset(
            data=train_data,
            categorical_features=categorical_features,
            target_column=config.data.caller.target_column,
            max_len=config.data.max_len,
            tokenizer=config.model.tokenizer
        )
        val_dataset = TBIDataset(
            data=val_data,
            categorical_features=categorical_features,
            target_column=config.data.caller.target_column,
            max_len=config.data.max_len,
            tokenizer=config.model.tokenizer
        )
        # ==========================================================Create dataloaders========================================
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        )
        # ==========================================================Create model========================================
        model = hydra.utils.instantiate(config.model)
        # ==========================================================Create optimizer========================================
        optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())
        # ==========================================================Create scheduler========================================        
        # =====================================================Create dataloaders====================================
    except Exception as e:
        import traceback
        traceback.print_exc()  # In ra stacktrace đầy đủ


if __name__ == "__main__":
    train()
