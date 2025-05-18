import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import os
import logging
from datetime import datetime

from models.models import (
    MediatorModel, ClientModel1, ClientModel2, ClientModel3,
    ClientModel4, ClientModel5, ClientModel6, ClientModel7, ClientModel8
)
from config.default_config import DefaultConfig
from utils.data_utils import get_dataset, split_data, get_dataloader
from utils.train_utils import (
    train_epoch, evaluate, compute_forgetting_degree,
    knowledge_distillation, aggregate_models
)

class FedBKT:
    """FedBKT: Federated Learning with Bidirectional Knowledge Transfer."""
    
    def __init__(self, config: DefaultConfig):
        """Initialize FedBKT.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize datasets and dataloaders
        self._setup_data()
        
        # Initialize models
        self._setup_models()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        log_file = os.path.join(
            self.config.log_dir,
            f"fedbkt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def _setup_data(self):
        """Setup datasets and dataloaders."""
        # Get datasets
        train_dataset, test_dataset = get_dataset(self.config)
        
        # Split data among clients
        self.client_datasets = split_data(
            train_dataset,
            self.config.num_clients,
            non_iid=True
        )
        
        # Create dataloaders
        self.client_dataloaders = [
            get_dataloader(dataset, self.config)
            for dataset in self.client_datasets
        ]
        self.test_dataloader = get_dataloader(test_dataset, self.config)
        
    def _setup_models(self):
        """Initialize client models and mediator model."""
        # Initialize client models with different architectures
        model_classes = [
            ClientModel1, ClientModel2, ClientModel3, ClientModel4,
            ClientModel5, ClientModel6, ClientModel7, ClientModel8
        ]
        
        # Assign models to clients in a round-robin fashion
        self.client_models = []
        for i in range(self.config.num_clients):
            model_class = model_classes[i % len(model_classes)]
            self.client_models.append(model_class(self.config.num_classes).to(self.device))
        
        # Initialize mediator model
        self.mediator_model = MediatorModel(self.config.num_classes).to(self.device)
        
        # Initialize optimizers
        self.client_optimizers = [
            optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
            for model in self.client_models
        ]
        self.mediator_optimizer = optim.SGD(
            self.mediator_model.parameters(),
            lr=self.config.mediator_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
    def train(self):
        """Main training loop following the FedBKT algorithm."""
        logging.info("Starting FedBKT training...")
        
        for round_idx in range(self.config.num_rounds):
            logging.info(f"Round {round_idx + 1}/{self.config.num_rounds}")
            
            # Step 1: Local Training
            client_metrics = self._local_training()
            
            # Step 2: Knowledge Extraction
            extracted_knowledge = self._knowledge_extraction()
            
            # Step 3: Mediator Update
            mediator_metrics = self._mediator_update(extracted_knowledge)
            
            # Step 4: Knowledge Sharing
            self._knowledge_sharing()
            
            # Log metrics
            self._log_metrics(round_idx, client_metrics, mediator_metrics)
            
            # Save checkpoints
            if (round_idx + 1) % self.config.save_interval == 0:
                self._save_checkpoint(round_idx)
                
        logging.info("Training completed!")
        
    def _local_training(self) -> List[Dict[str, float]]:
        """Local training phase for each client.
        
        Returns:
            List of client training metrics
        """
        client_metrics = []
        
        for client_idx, (model, optimizer, dataloader) in enumerate(
            zip(self.client_models, self.client_optimizers, self.client_dataloaders)
        ):
            # Store old model state for forgetting degree computation
            old_model = ClientModel1(self.config.num_classes).to(self.device)
            old_model.load_state_dict(model.state_dict())
            
            # Local training
            for epoch in range(self.config.local_epochs):
                metrics = train_epoch(model, dataloader, optimizer, self.device)
                
            # Compute forgetting degree
            forgetting_degree = compute_forgetting_degree(
                old_model, model, dataloader, self.device
            )
            
            metrics['forgetting_degree'] = forgetting_degree
            client_metrics.append(metrics)
            
        return client_metrics
        
    def _knowledge_extraction(self) -> List[Dict[str, torch.Tensor]]:
        """Extract knowledge from client models.
        
        Returns:
            List of extracted knowledge from each client
        """
        extracted_knowledge = []
        
        for client_idx, (model, dataloader) in enumerate(
            zip(self.client_models, self.client_dataloaders)
        ):
            model.eval()
            features_list = []
            logits_list = []
            
            with torch.no_grad():
                for inputs, _ in dataloader:
                    inputs = inputs.to(self.device)
                    features = model.get_features(inputs)
                    logits = model.get_logits(inputs)
                    features_list.append(features)
                    logits_list.append(logits)
                    
            extracted_knowledge.append({
                'features': torch.cat(features_list, dim=0),
                'logits': torch.cat(logits_list, dim=0)
            })
            
        return extracted_knowledge
        
    def _mediator_update(self, extracted_knowledge: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Update mediator model using extracted knowledge.
        
        Args:
            extracted_knowledge: List of extracted knowledge from clients
            
        Returns:
            Mediator training metrics
        """
        self.mediator_model.train()
        total_loss = 0
        total_l1_loss = 0
        total_l2_loss = 0
        total_l3_loss = 0
        
        for client_idx, knowledge in enumerate(extracted_knowledge):
            self.mediator_optimizer.zero_grad()
            
            # Get forgetting degree for adaptive weight
            forgetting_degree = compute_forgetting_degree(
                self.client_models[client_idx],
                self.mediator_model,
                self.client_dataloaders[client_idx],
                self.device
            )
            
            # Compute knowledge distillation loss
            losses = knowledge_distillation(
                self.mediator_model,
                self.client_models[client_idx],
                knowledge['features'],
                knowledge['logits'],
                self.config.knowledge_temp,
                forgetting_degree
            )
            
            # Update mediator model
            losses['total_loss'].backward()
            self.mediator_optimizer.step()
            
            total_loss += losses['total_loss'].item()
            total_l1_loss += losses['l1_loss'].item()
            total_l2_loss += losses['l2_loss'].item()
            total_l3_loss += losses['l3_loss'].item()
            
        return {
            'loss': total_loss / len(extracted_knowledge),
            'l1_loss': total_l1_loss / len(extracted_knowledge),
            'l2_loss': total_l2_loss / len(extracted_knowledge),
            'l3_loss': total_l3_loss / len(extracted_knowledge)
        }
        
    def _knowledge_sharing(self):
        """Share knowledge from mediator to clients."""
        self.mediator_model.eval()
        
        for client_idx, (model, dataloader) in enumerate(
            zip(self.client_models, self.client_dataloaders)
        ):
            model.train()
            
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Get forgetting degree for adaptive weight
                forgetting_degree = compute_forgetting_degree(
                    model,
                    self.mediator_model,
                    dataloader,
                    self.device
                )
                
                # Compute knowledge distillation loss
                losses = knowledge_distillation(
                    model,
                    self.mediator_model,
                    inputs,
                    targets,
                    self.config.knowledge_temp,
                    forgetting_degree
                )
                
                # Update client model
                self.client_optimizers[client_idx].zero_grad()
                losses['total_loss'].backward()
                self.client_optimizers[client_idx].step()
                
    def _log_metrics(self, round_idx: int,
                    client_metrics: List[Dict[str, float]],
                    mediator_metrics: Dict[str, float]):
        """Log training metrics.
        
        Args:
            round_idx: Current round index
            client_metrics: List of client metrics
            mediator_metrics: Mediator metrics
        """
        avg_client_accuracy = np.mean([m['accuracy'] for m in client_metrics])
        avg_client_loss = np.mean([m['loss'] for m in client_metrics])
        avg_forgetting = np.mean([m['forgetting_degree'] for m in client_metrics])
        
        logging.info(
            f"Round {round_idx + 1} - "
            f"Client Accuracy: {avg_client_accuracy:.4f}, "
            f"Client Loss: {avg_client_loss:.4f}, "
            f"Forgetting Degree: {avg_forgetting:.4f}, "
            f"Mediator Loss: {mediator_metrics['loss']:.4f}"
        )
        
    def _save_checkpoint(self, round_idx: int):
        """Save model checkpoints.
        
        Args:
            round_idx: Current round index
        """
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Save client models
        for client_idx, model in enumerate(self.client_models):
            torch.save(
                model.state_dict(),
                os.path.join(
                    self.config.save_dir,
                    f"client_{client_idx}_round_{round_idx + 1}.pt"
                )
            )
            
        # Save mediator model
        torch.save(
            self.mediator_model.state_dict(),
            os.path.join(
                self.config.save_dir,
                f"mediator_round_{round_idx + 1}.pt"
            )
        )

if __name__ == '__main__':
    config = DefaultConfig()
    fedbkt = FedBKT(config)
    fedbkt.train() 