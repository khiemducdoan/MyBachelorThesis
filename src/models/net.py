import torch
import torch.nn as nn
from typing import Dict, Any
from pytorch_tabnet.tab_network import TabNet
from src.models.base import BaseModel

from src.data.dataset import TBIDataset
from torch.utils.data import DataLoader
import pandas as pd


class Net(BaseModel):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params
        self.network = nn.Sequential(
            nn.Linear(params['D_in'], params['H']),
            nn.ReLU(),
            nn.Linear(params['H'], params['D_out'])
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

def main():
    data = pd.read_csv('/home/khanhnt/Khiem/MyBachelorThesis/dataset/raw/dataset.csv')
    config = {
        'target_column': 'd_kl_tl'
    }
    params = {
        'D_in': 10,
        'D_out': 4,
        'H': 15
    }
    model = Net(params)
    print(model)
    dataset = TBIDataset(data, config)
    print(dataset[0])
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(model(train_loader[0][0]))

if __name__ == '__main__':
    main()