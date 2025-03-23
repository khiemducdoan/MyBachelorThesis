import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Optional

class TBIDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        config: Dict,
        transform: Optional[callable] = None
    ):
        self.data = data
        self.config = config
        self.transform = transform
        
        # Separate features and target
        self.features = self.data.drop(columns=[config['target_column']])
        self.targets = self.data[config['target_column']]
        
        # # Get feature indices
        # self.num_indices = [self.features.columns.get_loc(col) 
        #                    for col in config['features']['numerical']]
        # self.cat_indices = [self.features.columns.get_loc(col) 
        #                    for col in config['features']['categorical']]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x = self.features.iloc[idx].values
        y = self.targets.iloc[idx]
        
        # Convert to tensor
        x = torch.FloatTensor(x)
        y = torch.LongTensor([y])[0]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y 
    
def main():
    data = pd.read_csv('/home/khanhnt/Khiem/MyBachelorThesis/dataset/raw/dataset.csv')
    config = {
        'target_column': 'd_kl_tl'
    }
    dataset = TBIDataset(data, config)
    print(dataset[0])

if __name__ == '__main__':
    main()