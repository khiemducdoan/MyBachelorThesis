import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Optional
import transformers

class TBIDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        num_classes: int,
        num_features: 64,
        transform: Optional[callable] = None
    ):
        self.data = data
        self.target_column = target_column
        self.num_classes = num_classes
        self.num_features = num_features
        self.transform = transform
        
        # Separate features and target
        self.features = self.data.drop(columns=[self.target_column])
        self.targets = self.data[self.target_column]
        
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
    
class ViTBERT(Dataset):
    def __init__(self, data : pd.DataFrame,
                 tokenizer: str,
                 target_column: str,
                 transform: Optional[callable] = None):
        self.data = data
        #only get dataset in index, index is taken from train ids by kfold from sklearn
        #if type = "train", augment dataset 
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]  # Extract text from dataframe
        label = self.data.iloc[idx, 1]  # Extract label

        # Tokenization
        tokenized_text = self.tokenizer(text, 
                                        add_special_tokens=True, 
                                        max_length=100, 
                                        padding='max_length', 
                                        truncation=True, 
                                        return_tensors='pt')

        # Extract fields from tokenized text
        input_ids = tokenized_text['input_ids'].squeeze()
        attention_mask = tokenized_text['attention_mask'].squeeze()

        # Convert label to tensor
        label = torch.tensor(label - 1, dtype=torch.long)  # Changed dtype to torch.long

        # Return tensors
        return (input_ids, attention_mask) , label  # Return attention_mask

    def __len__(self):
        return len(self.data)
def main():
    dataset = ViTBERT(data, tokenizer='bert-base-uncased')
    
    for i in range(len(dataset)):
        print(dataset[i])

if __name__ == "__main__":
    main()
