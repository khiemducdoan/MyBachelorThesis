import math
import torch
from torch import Tensor
from torch.nn import Sigmoid
import torch.nn.functional as F
from src.models.tabular_tokenizer import CategoricalFeatureTokenizer
from src.models.base import BaseModel

from transformers import LongformerModel

# from tabular_tokenizer import CategoricalFeatureTokenizer
# from base import BaseModel


from typing import Tuple, Optional
from transformers import AutoModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import hydra
from typing import Union
import torch.nn as nn

__all__ = ["NAIMclassifier"]
# def set_naim_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
#     """
#     Set the parameters of the NAIM model
#     Parameters
#     ----------
#     model_cfg : dict
#     preprocessing_params : Union[dict, DictConfig]
#     train_set : SupervisedTabularDatasetTorch

#     Returns
#     -------
#     dict
#         model_cfg
#     """
#     model_cfg["init_params"]["input_size"] = train_set.input_size
#     model_cfg["init_params"]["output_size"] = train_set.output_size

#     cat_idxs, cat_dims = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)
#     model_cfg["init_params"]["cat_idxs"] = cat_idxs
#     model_cfg["init_params"]["cat_dims"] = cat_dims

#     searched_value, key_found = recursive_cfg_search(model_cfg, "d_token")
#     if searched_value is None and key_found:
#         d_token, _ = recursive_cfg_search(model_cfg, "input_size")
#         log.info(f"Token dimension: {d_token}")
#         model_cfg = recursive_cfg_substitute(model_cfg, {"d_token": d_token})

#     searched_value, key_found = recursive_cfg_search(model_cfg, "num_heads")
#     if searched_value is None and key_found:
#         divisor = 2
#         log.info(f"N. attention heads: {divisor}")
#         model_cfg = recursive_cfg_substitute(model_cfg, {"num_heads": divisor})

#     return model_cfg
# def compute_categorical_idxs_dims(columns: list, 
#                                   preprocessing_params: Union[dict, DictConfig]) -> Tuple[list, list]:
#     """
#     Compute the indexes and dimensions of the categorical features in the dataset
#     Parameters
#     ----------
#     columns : list
#     preprocessing_params : Union[dict, DictConfig]

#     Returns
#     -------
#     Tuple[list, list]
#         categorical_idxs, categorical_dims
#     """
#     unique_values = preprocessing_params.categorical_unique_values
#     categorical_columns = tuple(unique_values.index.to_list())

#     categorical_idxs = []
#     categorical_dims = []
#     for idx, col in enumerate(columns):
#         if col in categorical_columns:
#             categorical_idxs.append(idx)
#             categorical_dims.append(len(unique_values[col]))
#         elif col.split("_")[0] in categorical_columns:
#             categorical_idxs.append(idx)
#             categorical_dims.append(2)
#     return categorical_idxs, categorical_dims
class Classifier(nn.Module):
    def __init__(self, config, dropout_rate=0.1):
        super().__init__()

        self.dropout_1 = nn.Dropout(dropout_rate*2)
        self.dense_1  = nn.Linear(config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dense_2 = nn.Linear(128,4)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, feature):

        feature = self.dropout_1(feature)
        feature = self.dense_1(feature)
        feature = self.relu(feature)
        feature = self.dropout_2(feature)
        feature = self.dense_2(feature)

        feature = self.sigmoid(feature)
        return feature
class DynamicClassifier(nn.Module):
    def __init__(self, 
                input_dim=768, 
                num_classes=4, 
                dropout_rate=0.25, 
                num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            output_dim = input_dim
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, num_classes))
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, feature):
        return self.model(feature)

class ViTBERTClassifier(nn.Module):
    def __init__(self, 
                pretrained_model_name, 
                num_classes=4, 
                dropout_rate=0.25, 
                num_layers=4):
        super().__init__()
        self.bert = LongformerModel.from_pretrained(pretrained_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dynamic_classifier = DynamicClassifier(input_dim=self.bert.config.hidden_size,
                                            num_classes=num_classes,
                                            dropout_rate=dropout_rate,
                                            num_layers=num_layers)
        self.static_classifier = Classifier(self.bert.config, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Always available: last_hidden_state
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Option 1: CLS token (rất phổ biến)
        cls_token = last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)

        # Option 2: Mean pooling (ít sensitive hơn CLS)
        # mean_pool = last_hidden_state.mean(dim=1)

        # Lưu ý: static_classifier đầu vào phải là (B, hidden_dim)
        logits = self.static_classifier(cls_token)
        return logits
class TabularMasker:
    def __init__(self, mask_type: int = 0, missing_value: str = "-inf"):
        self.mask_type = mask_type
        missing_value_options = {"-inf": -torch.inf, "~inf": -1e9}
        self.missing_value = missing_value_options[missing_value]

    def _tabular_sample_mask(self, sample: Tensor):
        mask = torch.clone(sample)
        mask[~torch.isnan(sample)] = 0
        mask[torch.isnan(sample)] = 1
        return mask

    def mask(self, data: Tensor):
        masks = Tensor().to(data.device)

        for sample in data:
            sample_mask = self._tabular_sample_mask(sample).to(torch.bool)

            sample_mask = sample_mask.repeat(sample_mask.shape[0], 1)

            if self.mask_type == 1:
                sample_mask = (~sample_mask)
                sample_mask = ~(sample_mask.mul(sample_mask.T).to(torch.bool))

            masks = torch.cat([masks, sample_mask.unsqueeze(dim=0)], dim=0)

        masks = torch.masked_fill(masks, masks.to(torch.bool), self.missing_value)

        if self.mask_type != 2:
            return masks, None

        return masks, masks.transpose(-2, -1)


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention module.
    """
    def __init__(self,
                 input_size: int,
                 num_heads: int,
                 bias: bool = True,
                 activation: str = "relu",
                 dropout_rate: float = 0.0):

        super(MultiHeadAttention, self).__init__()
        assert input_size % num_heads == 0, f"`input_size`({input_size}) should be divisible by `num_heads`({num_heads})"

        self.input_size = input_size
        self.num_heads = num_heads
        self.bias = bias
        activation_options = dict(relu=F.relu, gelu=F.gelu, tanh=F.tanh)
        self.activation = activation_options[activation]
        self.dropout_rate = dropout_rate

        self.linear_q = torch.nn.Linear(input_size, input_size, bias)
        self.linear_k = torch.nn.Linear(input_size, input_size, bias)
        self.linear_v = torch.nn.Linear(input_size, input_size, bias)
        self.linear_o = torch.nn.Linear(input_size, input_size, bias)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None, mask2: Tensor = None) -> Tensor:
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        if mask is not None:
            mask = torch.repeat_interleave(mask, self.num_heads, 0 )
        if mask2 is not None:
            mask2 = torch.repeat_interleave(mask2, self.num_heads, 0 )

        y, attn_scores = self._scaled_dot_product_attention(q, k, v, attn_mask=mask, attn_mask_2=mask2)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def _reshape_to_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def _scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None, attn_mask_2: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        B, Nt, E = q.shape
        q = q / math.sqrt(E)

        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
        else:
            attn = torch.bmm(q, k.transpose(-2, -1))

        attn = F.softmax(attn, dim=-1)

        if attn_mask_2 is not None:
            attn = torch.add(attn, attn_mask_2)
            attn = F.relu(attn)

        if self.dropout_rate > 0.0:
            attn = F.dropout(attn, p=self.dropout_rate)

        output = torch.bmm(attn, v)

        return output, attn


class EncoderBlock(torch.nn.Module):
    """
    Encoder block of the Transformer.
    """
    def __init__(self, emb_dim, ff_dim, num_heads, bias: bool = False, activation: str = "relu", dropout_rate: float = 0.0):
        super(EncoderBlock, self).__init__()

        self.layer_norm_1 = torch.nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads, bias=bias, activation=activation, dropout_rate=dropout_rate )
        self.layer_norm_2 = torch.nn.LayerNorm(emb_dim)

        activation_options = dict( relu= torch.nn.ReLU, gelu= torch.nn.GELU )
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, ff_dim),
            activation_options[activation](),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(ff_dim, emb_dim),
            torch.nn.Dropout(dropout_rate)
        )

    def forward(self, x: Tensor, mask: Tensor = None, mask2: Tensor = None):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x, mask=mask, mask2=mask2)
        x = self.layer_norm_2(x)
        x = x + self.ff(x)
        return x


class NAIM(torch.nn.Module):
    """
    NAIM model for tabular data.
    """
    def __init__(self,
                 input_size,
                 output_size,
                 cat_idxs: list,
                 cat_dims: list,
                 d_token: int,
                 embedder_initialization: str,
                 bias: bool,
                 mask_type: int = 0,
                 missing_value: str = "-inf",
                 num_heads: int = 12,
                 feedforward_dim: int = 1000,
                 dropout_rate: float = 0.1,
                 activation: str = "relu",
                 num_layers: int = 12,
                 extractor: bool = False):

        super(NAIM, self).__init__()

        self.input_size = input_size
        if extractor:
            self.output_size = input_size * d_token
        else:
            self.output_size = output_size
        self.cat_idxs = cat_idxs if cat_idxs else [-1]
        self.cat_dims = cat_dims if cat_dims else [-1]
        self.d_token = d_token
        self.embedder_initialization = embedder_initialization
        self.bias = bias
        self.mask_type = mask_type
        self.missing_value = missing_value
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.num_layers = num_layers
        self.extractor = extractor

        # EMBEDDERS initializations
        j = 0
        self.embeddings = torch.nn.ModuleList()
        common_params = dict( d_token=self.d_token, bias=self.bias, initialization=self.embedder_initialization )
        for i in range(input_size):
            is_categorical_feature = i in self.cat_idxs
            feature_type_params = { True: dict(cardinalities = [self.cat_dims[j] + 1], padding_idx=self.cat_dims[j]),
                                    False: dict(cardinalities = [2], padding_idx=1) }

            j = j + (is_categorical_feature * (i != self.cat_idxs[-1]))
            embedding = CategoricalFeatureTokenizer( **common_params, **feature_type_params[ is_categorical_feature ])

            self.embeddings.append(embedding)

        # MASKER initialization
        self.attention_mask = TabularMasker(self.mask_type, self.missing_value)

        self.dropout = torch.nn.Dropout(self.dropout_rate)

        self.encoder = torch.nn.ModuleList([EncoderBlock(self.d_token, self.feedforward_dim, self.num_heads, bias=self.bias, activation=self.activation, dropout_rate=self.dropout_rate) for _ in range(self.num_layers)])

        self.norm = torch.nn.LayerNorm(self.d_token)

        # classifier
        if not self.extractor:
            if self.output_size > 1:
                self.classifier = torch.nn.Sequential(torch.nn.Linear(self.input_size*self.d_token, self.output_size))
            else:
                self.classifier = torch.nn.Sequential(torch.nn.Linear(self.input_size*self.d_token, self.output_size), Sigmoid())

    def forward(self, x):

        j = 0
        embeddings = Tensor().to(x.device)
        for feature_idx in list(range(x.shape[1])):
            if feature_idx in self.cat_idxs:
                single_feature = torch.nan_to_num(x[:, feature_idx], nan=self.cat_dims[j]).to(torch.int64)
                feature_values = None
                j += 1
            else:
                single_feature = torch.isnan(x[:, feature_idx]).to(torch.int64)
                feature_values = torch.nan_to_num(x[:, feature_idx], nan=0)
            single_feature_embedding = self.embeddings[feature_idx](single_feature, feature_values)

            single_feature_embedding = torch.swapaxes(single_feature_embedding, 0, 1)
            embeddings = torch.cat([embeddings, single_feature_embedding], dim=1)

        masks, masks2 = self.attention_mask.mask(x)

        # transformer
        for encoder_layer in self.encoder:
            residual = embeddings
            embeddings = encoder_layer(embeddings, mask=masks, mask2=masks2)
            embeddings = embeddings + residual  # residual connection

        embeddings = self.norm(embeddings)

        features = embeddings.view(embeddings.shape[0], -1)

        if self.extractor:
            return features

        # classifier
        logits = self.classifier(features)

        return logits
    
class NAIMclassifier(BaseModel):
    def __init__(self, params):
        super(NAIMclassifier, self).__init__()
        self.naim = NAIM(**params)

    def forward(self, x):
        return self.naim(x)
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
    def configure_loss(self, config):
        if config.loss.weight == 0:
            loss_fn = hydra.utils.instantiate(config.loss, weight = None)

        else:
            weight_tensor = torch.tensor(config.loss.weight, dtype=torch.float32)
            weight_tensor = weight_tensor.to(next(self.parameters()).device)
            loss_fn = hydra.utils.instantiate(config.loss, weight=weight_tensor)
        return loss_fn
    
class NAIM_TEXTclassifier(BaseModel):
    def __init__(self, params_naim, params_vibert):
        super(NAIM_TEXTclassifier, self).__init__()
        self.naim = NAIM(**params_naim)
        self.vitbi = ViTBERTClassifier(**params_vibert)
    def forward(self, x, input_ids, attention_mask=None):
        clinical_features = self.naim(x)
        text_features = self.vitbi(input_ids, attention_mask)
        #add clinical features to text features
        final = clinical_features + text_features
        return final
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
    def configure_loss(self, config):
        if config.loss.weight == 0:
            loss_fn = hydra.utils.instantiate(config.loss, weight = None)

        else:
            weight_tensor = torch.tensor(config.loss.weight, dtype=torch.float32)
            weight_tensor = weight_tensor.to(next(self.parameters()).device)
            loss_fn = hydra.utils.instantiate(config.loss, weight=weight_tensor)
        return loss_fn

def main():
    params_naim = {
    "input_size": 64,
    "output_size": 4,  # bỏ qua nếu extractor=True
    "cat_idxs": [],  # không có feature dạng category
    "cat_dims": [],
    "d_token": 32,
    "embedder_initialization": "normal",
    "bias": True,
    "mask_type": 0,
    "missing_value": "-inf",
    "num_heads": 4,
    "feedforward_dim": 256,
    "dropout_rate": 0.1,
    "activation": "relu",
    "num_layers": 2,
    "extractor": True  # Set True để trích đặc trưng
    }
    values= ["emilyalsentzer/Bio_ClinicalBERT",
    "dmis-lab/biobert-base-cased-v1.1",
    "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "nvidia/biomegatron-345m-uncased",
    "yikuan8/Clinical-Longformer"
    ]  # 
    params_vibert = {
        "pretrained_model_name": "vinai/phobert-base",
        "num_classes": 4,  # bỏ qua nếu extractor
        "dropout_rate": 0.1,
        "num_layers": 2
    }

    # === Step 1: Khởi tạo mô hình ===
    naim = NAIMclassifier(params=params_naim)
    vibert = ViTBERTClassifier(**params_vibert)
    # === Step 2: Tạo dữ liệu mẫu ===
    batch_size = 2

    # Giả lập tabular data: (B, 64)
    table_data = torch.rand((batch_size, 64))

    # Giả lập văn bản: (B, L)
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    sample_texts = ["bệnh nhân nhập viện trong tình trạng đau đầu dữ dội.", "tiểu sử tiểu đường lâu năm, đã điều trị thuốc."]
    encoded_inputs = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")

    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    # === Step 3: Chạy mô hình ===
    with torch.no_grad():
        output_naim = naim(table_data)
        output_vibert = vibert(input_ids, attention_mask)

    # === Step 4: In thông tin ===
    print("Table data shape:", table_data.shape)
    print("Text input_ids shape:", input_ids.shape)
    print("Output_naim shape:", output_naim.shape)
    print("Output_vibert shape:", output_vibert.shape)

    # Nếu NAIM là extractor:
    naim_feature_shape = table_data.shape[0], params_naim["input_size"] * params_naim["d_token"]
    print("Expected NAIM output (extractor):", naim_feature_shape)

if __name__ == "__main__":
    main()
