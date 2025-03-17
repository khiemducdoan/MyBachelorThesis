from typing import Dict, Any
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Base class for all models"""
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass logic"""
        raise NotImplementedError
        
    @abstractmethod
    def configure_optimizers(self, config: Dict[str, Any]):
        """Configure optimizer and scheduler"""
        raise NotImplementedError
        
    def save(self, path: str):
        """Save model state"""
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        """Load model state"""
        self.load_state_dict(torch.load(path)) 