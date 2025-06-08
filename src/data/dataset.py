import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Optional
import transformers

class TBIDataset(Dataset):
    def __init__(
        self,
        data,
        target_column: str,
        num_classes: int,
        num_features: 64,
        categorical_features: Optional[callable] = None,
        transform: Optional[callable] = None
    ):
        self.data = data
        self.target_column = target_column
        self.num_classes = num_classes
        self.num_features = num_features
        self.transform = transform
        # Separate features and target
        self.features = self.data.drop( columns=[self.target_column, "text"]) 
        self.targets = self.data[self.target_column] -1
        self.features_categorical = self.data[categorical_features] if categorical_features else None
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
            x = x.apply(lambda x: (x - x.mean()) / x.std())
            
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
                                        max_length=512, 
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
    


# class TBIDatasetDualStream(Dataset):
#     def __init__(
#         self,
#         data,
#         target_column: str,
#         num_classes: int,
#         num_features: 64,
#         categorical_features: Optional[callable] = None,
#         transform: Optional[callable] = None
#     ):
#         self.data = data
#         self.target_column = target_column
#         self.num_classes = num_classes
#         self.num_features = num_features
#         self.transform = transform
#         # Separate features and target
#         self.features = self.data.drop(columns=[self.target_column])
#         self.targets = self.data[self.target_column]
#         self.features_categorical = self.data[categorical_features]
        
#     def __len__(self):
#         return len(self.data)
        
#     def __getitem__(self, idx):
#         x = self.features.iloc[idx].values
#         y = self.targets.iloc[idx] - 1
        
#         # Create mask for missing data
#         mask = np.where(pd.isnull(x), 1, 0)
        
#         # Replace missing values with 0
#         x = np.nan_to_num(x)
        
#         # Convert to tensor
#         x = torch.FloatTensor(x)
#         mask = torch.FloatTensor(mask)
#         y = torch.LongTensor([y])[0]
        
#         if self.transform:
#             x = self.transform(x)
            
#         return (x, mask), y
# def main():
#     dataset = ViTBERT(data, tokenizer='tokenizer-base-uncased')
    
#     for i in range(len(dataset)):
#         print(dataset[i])

# if __name__ == "__main__":
#     main()



class TBIDataset2stream(Dataset):
    def __init__(
        self,
        data,
        target_column: str,
        num_classes: int,
        num_features: 64,
        tokenizer: str,
        categorical_features: Optional[callable] = None,
        transform: Optional[callable] = None
    ):
        self.data = data
        self.target_column = target_column
        self.num_classes = num_classes
        self.num_features = num_features
        self.transform = transform
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        # Separate features and target
        self.features = self.data.drop(columns=[self.target_column, "text"]) 
        self.text = self.data["text"] 
        self.targets = self.data[self.target_column] -1
        self.features_categorical = self.data[categorical_features] if categorical_features else None
        # # Get feature indices
        # self.num_indices = [self.features.columns.get_loc(col) 
        #                    for col in config['features']['numerical']]
        # self.cat_indices = [self.features.columns.get_loc(col) 
        #                    for col in config['features']['categorical']]
        
    def __len__(self):
        return len(self.data)
    def read_text(self, idx):
        text = self.text.iloc[idx]
        if pd.isnull(text):
            # Trả về tensor zeros nếu không có text
            input_ids = torch.zeros(512, dtype=torch.long)
            attention_mask = torch.zeros(512, dtype=torch.long)
            return input_ids, attention_mask
        tokenized_text = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokenized_text['input_ids'].squeeze(0)
        attention_mask = tokenized_text['attention_mask'].squeeze(0)
        return input_ids, attention_mask
    def __getitem__(self, idx):
        x = self.features.iloc[idx].values
        y = self.targets.iloc[idx]
        input_ids, attention_mask = self.read_text(idx)
        # Convert to tensor
        x = torch.FloatTensor(x)
        y = torch.LongTensor([y])[0]
        
        if self.transform:
            x = x.apply(lambda x: (x - x.mean()) / x.std())
            
        return (x,input_ids, attention_mask), y 