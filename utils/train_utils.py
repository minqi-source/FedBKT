import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def train_epoch(model: nn.Module, 
                dataloader: torch.utils.data.DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device) -> Dict[str, float]:
    """Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        metrics = model.compute_metrics(outputs, targets)
        correct += metrics['correct']
        total += metrics['total']
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total
    }

def evaluate(model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            device: torch.device) -> Dict[str, float]:
    """Evaluate model.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = model.compute_loss(outputs, targets)
            
            total_loss += loss.item()
            metrics = model.compute_metrics(outputs, targets)
            correct += metrics['correct']
            total += metrics['total']
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total
    }

def compute_forgetting_degree(old_model: nn.Module,
                             new_model: nn.Module,
                             dataloader: torch.utils.data.DataLoader,
                             device: torch.device) -> float:
    """Compute forgetting degree between old and new model.
    
    Args:
        old_model: Old model state
        new_model: New model state
        dataloader: DataLoader for computing forgetting degree
        device: Device to compute on
        
    Returns:
        Forgetting degree value
    """
    old_model.eval()
    new_model.eval()
    
    total_diff = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            old_outputs = old_model.get_features(inputs)
            new_outputs = new_model.get_features(inputs)
            
            # Compute L2 distance between feature representations
            diff = torch.norm(old_outputs - new_outputs, dim=1).mean()
            total_diff += diff.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    return total_diff / total_samples

def knowledge_distillation(student_model: nn.Module,
                          teacher_model: nn.Module,
                          inputs: torch.Tensor,
                          targets: torch.Tensor,
                          temperature: float,
                          forgetting_degree: float = None) -> Dict[str, torch.Tensor]:
    """Compute knowledge distillation loss following the FedBKT framework.
    
    Args:
        student_model: Student model
        teacher_model: Teacher model
        inputs: Input tensor
        targets: Target labels
        temperature: Temperature for softmax
        forgetting_degree: Forgetting degree for adaptive weight (optional)
        
    Returns:
        Dictionary containing different loss components
    """
    # L1: KL divergence between soft labels
    with torch.no_grad():
        teacher_logits = teacher_model.get_logits(inputs)
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=1)
    
    student_logits = student_model.get_logits(inputs)
    student_probs = torch.softmax(student_logits / temperature, dim=1)
    l1_loss = -(teacher_probs * torch.log(student_probs)).sum(dim=1).mean()
    
    # L2: Feature-level knowledge transfer
    teacher_features = teacher_model.get_features(inputs)
    student_features = student_model.get_features(inputs)
    
    # If forgetting degree is provided, use adaptive weight
    if forgetting_degree is not None:
        delta = torch.exp(-forgetting_degree)
        l2_loss = delta * F.mse_loss(student_features, teacher_features)
    else:
        l2_loss = F.mse_loss(student_features, teacher_features)
    
    # L3: Cross-entropy loss with ground truth
    l3_loss = F.cross_entropy(student_logits, targets)
    
    return {
        'l1_loss': l1_loss,
        'l2_loss': l2_loss,
        'l3_loss': l3_loss,
        'total_loss': l1_loss + l2_loss + l3_loss
    }

def aggregate_models(models: List[nn.Module],
                    weights: List[float] = None) -> Dict[str, torch.Tensor]:
    """Aggregate multiple models using weighted averaging.
    
    Args:
        models: List of models to aggregate
        weights: List of weights for each model (optional)
        
    Returns:
        Dictionary of aggregated model parameters
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    aggregated_params = {}
    for param_name in models[0].state_dict().keys():
        aggregated_params[param_name] = sum(
            model.state_dict()[param_name] * weight
            for model, weight in zip(models, weights)
        )
    
    return aggregated_params 