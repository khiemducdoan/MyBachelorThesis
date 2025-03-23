import torch
import torch.nn as nn
from typing import Dict, Any
from pytorch_tabnet.tab_network import TabNet
from src.models.base import BaseModel

from src.data.dataset import TBIDataset
from torch.utils.data import DataLoader
import pandas as pd


class TabNetClassifier(BaseModel):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params
        
        # self.network = TabNet(
        #     input_dim=params['input_size'],
        #     output_dim=params['output_size'],
        #     n_d=params['n_d'],
        #     n_a=params['n_a'],
        #     n_steps=params['n_steps'],
        #     gamma=params['gamma'],
        #     n_independent=params['n_independent'],
        #     n_shared=params['n_shared'],
        #     virtual_batch_size=params['virtual_batch_size'],
        #     momentum=params['momentum'],
        #     mask_type=params['mask_type']
        # )
        
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

def main():
    data = pd.read_csv('/home/khanhnt/Khiem/MyBachelorThesis/dataset/raw/dataset.csv')
    config = {
        'target_column': 'd_kl_tl'
    }
    params = {
        'input_size': 10,
        'output_size': 4,
        'n_d': 8,
        'n_a': 8,
        'n_steps': 3,
        'gamma': 1.3,
        'n_independent': 2,
        'n_shared': 2,
        'virtual_batch_size': 128,
        'momentum': 0.02,
        'mask_type': 'sparsemax'
    }
    model = TabNetClassifier(params)
    print(model)
    dataset = TBIDataset(data, config)
    print(dataset[0])
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(model(train_loader[0][0]))

if __name__ == '__main__':
    main()