import torch
import torch.nn as nn
from typing import Dict, Any

class BaseModel(nn.Module):
    """Base class for all models in FedBKT."""
    
    def __init__(self, config):
        """Initialize the base model.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        raise NotImplementedError
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the input.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        raise NotImplementedError
        
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get logits from the input.
        
        Args:
            x: Input tensor
            
        Returns:
            Logits tensor
        """
        raise NotImplementedError
        
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Loss value
        """
        return nn.CrossEntropyLoss()(outputs, targets)
        
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Compute evaluation metrics.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Dictionary of metrics
        """
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        
        return {
            'accuracy': correct / total,
            'correct': correct,
            'total': total
        } 