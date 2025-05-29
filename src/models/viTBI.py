import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.models.base import BaseModel
import hydra
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
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
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
        features_bert = outputs[0] # Last hidden state
        features_cls = features_bert[:, 0, :].unsqueeze(1) # CLS token
        pooled_output = outputs[1] # Pooled output
        # Access last_hidden_state (mandatory) and optionally pooler_output
        # last_hidden_state = outputs.last_hidden_state  # Always available
        # pooler_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
        # Use pooler_output if available; otherwise, use mean pooling
        # if pooler_output is not None:
        #     logits = self.classifier(pooler_output)
        # else:
        #     # Mean pooling over the last hidden state
        #     logits = self.classifier(last_hidden_state.mean(dim=1))
        logits = self.static_classifier(pooled_output)
        return logits

class viTBI(BaseModel):
    def __init__(self, params):
        super(viTBI, self).__init__()
        self.vitbi = ViTBERTClassifier(**params)

    def forward(self, input_ids, attention_mask=None):
        # input_ids, attention_mask = x
        return self.vitbi(input_ids, attention_mask)
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
        loss = hydra.utils.instantiate(config.loss)
        return loss
    def configure_loss(self, config):
        if config.loss.weight == 0:
            loss_fn = hydra.utils.instantiate(config.loss, weight = None)

        else:
            weight_tensor = torch.tensor(config.loss.weight, dtype=torch.float32)
            weight_tensor = weight_tensor.to(next(self.parameters()).device)
            loss_fn = hydra.utils.instantiate(config.loss, weight=weight_tensor)
        return loss_fn

# To use your modified classifier, you just need to specify the desired number of layers and dropout rate when initializing it:
def main():
    pass
if __name__ == "__main__":
    main()