import torch
import torch.nn as nn
from typing import Dict, Any
from pytorch_tabnet.tab_network import TabNet
from .base import BaseModel

class TabNetClassifier(BaseModel):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params
        
        self.network = TabNet(
            input_dim=params['input_size'],
            output_dim=params['output_size'],
            n_d=params['n_d'],
            n_a=params['n_a'],
            n_steps=params['n_steps'],
            gamma=params['gamma'],
            n_independent=params['n_independent'],
            n_shared=params['n_shared'],
            virtual_batch_size=params['virtual_batch_size'],
            momentum=params['momentum']
        )
        
    def forward(self, x):
        return self.network(x)
        
    def configure_optimizers(self, config):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config['scheduler']['mode'],
            factor=config['scheduler']['factor'],
            patience=config['scheduler']['patience']
        )
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'val_loss'
        } 