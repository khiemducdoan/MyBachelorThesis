from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from typing import Dict
from src.data.dataset import TBIDataset
from torch.utils.data import Dataset
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate various classification metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    } 

def set_naim_params(config,dataset :TBIDataset ,  **_):
    cat_idxs, cat_dims = compute_categorical_idxs_dims(dataset,dataset.features_categorical)
    config.model.params.cat_idxs = cat_idxs
    config.model.params.cat_dims = cat_dims
    return config

def compute_categorical_idxs_dims(df, categorical_features: list):
    """
    Compute categorical indices and dimensions from a DataFrame, ignoring NaN values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    categorical_features : list
        List of categorical column names.

    Returns
    -------
    Tuple[list, list]
        Categorical indices and dimensions (number of unique non-NaN values per categorical feature).
    """
    cat_idxs = []
    cat_dims = []

    for i, col in enumerate(df.features):
        if col not in categorical_features:
            continue
        cat_idxs.append(i)
        # Bỏ NaN trước khi lấy số lượng giá trị unique
        unique_vals = df[col].dropna().unique()
        cat_dims.append(len(unique_vals))

    return cat_idxs, cat_dims

