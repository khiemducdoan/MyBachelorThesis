import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
class DynamicClassifier(nn.Module):
    def __init__(self, 
                input_dim=768, 
                num_classes=4, 
                dropout_rate=0.25, 
                num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            output_dim = input_dim // 2
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.GELU())
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
        self.classifier = DynamicClassifier(input_dim=self.bert.config.hidden_size,
                                            num_classes=num_classes,
                                            dropout_rate=dropout_rate,
                                            num_layers=num_layers)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Access last_hidden_state (mandatory) and optionally pooler_output
        last_hidden_state = outputs.last_hidden_state  # Always available
        pooler_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else None

        # Use pooler_output if available; otherwise, use mean pooling
        if pooler_output is not None:
            logits = self.classifier(pooler_output)
        else:
            # Mean pooling over the last hidden state
            logits = self.classifier(last_hidden_state.mean(dim=1))
        
        return logits