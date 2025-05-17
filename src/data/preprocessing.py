import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, Tuple

class DataPreprocessor:
    def __init__(self, config: Dict):
        self.config = config
        self.num_scaler = StandardScaler()
        self.cat_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.num_imputer = SimpleImputer(strategy=config['preprocessing']['numerical']['missing_values'])
        self.cat_imputer = SimpleImputer(strategy=config['preprocessing']['categorical']['missing_values'])
        
    def fit(self, df: pd.DataFrame):
        num_features = self.config['features']['numerical']
        cat_features = self.config['features']['categorical']
        
        # Fit numerical features
        if num_features:
            num_data = df[num_features]
            self.num_imputer.fit(num_data)
            self.num_scaler.fit(self.num_imputer.transform(num_data))
            
        # Fit categorical features    
        if cat_features:
            cat_data = df[cat_features]
            self.cat_imputer.fit(cat_data)
            self.cat_encoder.fit(self.cat_imputer.transform(cat_data))
            
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        num_features = self.config['features']['numerical']
        cat_features = self.config['features']['categorical']
        result_dfs = []
        
        # Transform numerical features
        if num_features:
            num_data = df[num_features]
            num_imputed = self.num_imputer.transform(num_data)
            num_scaled = self.num_scaler.transform(num_imputed)
            result_dfs.append(pd.DataFrame(num_scaled, columns=num_features))
            
        # Transform categorical features    
        if cat_features:
            cat_data = df[cat_features]
            cat_imputed = self.cat_imputer.transform(cat_data)
            cat_encoded = self.cat_encoder.transform(cat_imputed)
            cat_columns = self.cat_encoder.get_feature_names_out(cat_features)
            result_dfs.append(pd.DataFrame(cat_encoded, columns=cat_columns))
            
        # Combine all features
        result = pd.concat(result_dfs, axis=1)
        
        return result 
    
def z_transform(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    num_features = config['features']['numerical']
    cat_features = config['features']['categorical']
    
    # Apply z-transform to numerical features
    if num_features:
        df[num_features] = (df[num_features] - df[num_features].mean()) / df[num_features].std()
        

    return df