import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from models.models import MediatorModel, ClientModel, MappingModule
from utils.utils import get_dataset, create_client_dataloaders, kl_divergence, calculate_forgetting_degree
from config.config import Config

class FedBKT:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 加载数据集
        self.trainset, self.testset, self.num_classes = get_dataset(
            config.dataset, config.data_dir)
        self.test_loader = DataLoader(self.testset, batch_size=config.batch_size)
        
        # 创建客户端数据加载器
        self.client_loaders = create_client_dataloaders(
            self.trainset, config.num_clients, config.alpha_dirichlet, config.batch_size)
        
        # 初始化模型
        self.mediator_model = MediatorModel(self.num_classes).to(self.device)
        self.client_models = []
        self.mapping_modules = []
        
        for i in range(config.num_clients):
            # 随机选择模型大小
            model_type = np.random.choice(['small', 'medium', 'large'])
            client_model = ClientModel(self.num_classes, model_type).to(self.device)
            self.client_models.append(client_model)
            
            # 创建映射模块
            if model_type == 'small':
                in_features = 128
            elif model_type == 'medium':
                in_features = 256
            else:
                in_features = 512
            mapping_module = MappingModule(in_features, 512).to(self.device)
            self.mapping_modules.append(mapping_module)
    
    def train_client(self, client_idx, round_idx):
        client_model = self.client_models[client_idx]
        mapping_module = self.mapping_modules[client_idx]
        client_loader = self.client_loaders[client_idx]
        
        # 优化器
        optimizer = optim.SGD([
            {'params': client_model.parameters()},
            {'params': mapping_module.parameters()}
        ], lr=self.config.learning_rate, momentum=self.config.momentum)
        
        client_model.train()
        mapping_module.train()
        
        for epoch in range(self.config.num_epochs):
            for batch_idx, (data, target) in enumerate(client_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 知识共享阶段
                with torch.no_grad():
                    mediator_output = self.mediator_model(data)
                
                client_output = client_model(data)
                mapped_features = mapping_module(client_model.fc1(client_model.features))
                
                # 计算损失
                l1 = kl_divergence(client_output, mediator_output, self.config.temperature)
                l2 = kl_divergence(mapped_features, self.mediator_model.fc1(self.mediator_model.features))
                l3 = nn.CrossEntropyLoss()(client_output, target)
                
                loss = l1 + l2 + l3
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 知识提取阶段
        self.mediator_model.train()
        optimizer_mediator = optim.SGD(self.mediator_model.parameters(),
                                     lr=self.config.learning_rate,
                                     momentum=self.config.momentum)
        
        for epoch in range(self.config.num_epochs):
            for batch_idx, (data, target) in enumerate(client_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 计算遗忘度
                if batch_idx == 0:
                    old_data = data
                else:
                    forgetting = calculate_forgetting_degree(
                        self.mediator_model, old_data, data, self.device)
                    old_data = data
                
                mediator_output = self.mediator_model(data)
                client_output = client_model(data)
                mapped_features = mapping_module(client_model.fc1(client_model.features))
                
                # 计算损失
                l1 = kl_divergence(mediator_output, client_output, self.config.temperature)
                l2 = nn.MSELoss()(mapped_features, self.mediator_model.fc1(self.mediator_model.features))
                l3 = nn.CrossEntropyLoss()(mediator_output, target)
                
                # 使用遗忘度调整权重
                delta = np.exp(-forgetting / (self.config.num_clients - 1))
                loss = delta * l1 + (1 - delta) * l2 + l3
                
                optimizer_mediator.zero_grad()
                loss.backward()
                optimizer_mediator.step()
        
        return self.mediator_model.state_dict()
    
    def aggregate_models(self, client_weights):
        # 加权平均聚合
        aggregated_weights = {}
        total_samples = sum(len(loader.dataset) for loader in self.client_loaders)
        
        for key in client_weights[0].keys():
            aggregated_weights[key] = torch.zeros_like(client_weights[0][key])
            for i, weights in enumerate(client_weights):
                weight = len(self.client_loaders[i].dataset) / total_samples
                aggregated_weights[key] += weight * weights
        
        self.mediator_model.load_state_dict(aggregated_weights)
    
    def evaluate(self):
        self.mediator_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.mediator_model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return 100. * correct / total
    
    def train(self):
        for round_idx in tqdm(range(self.config.num_rounds)):
            # 选择客户端
            selected_clients = np.random.choice(
                self.config.num_clients,
                size=max(1, int(self.config.num_clients * 0.1)),
                replace=False
            )
            
            # 客户端训练
            client_weights = []
            for client_idx in selected_clients:
                weights = self.train_client(client_idx, round_idx)
                client_weights.append(weights)
            
            # 模型聚合
            self.aggregate_models(client_weights)
            
            # 评估
            if (round_idx + 1) % 10 == 0:
                accuracy = self.evaluate()
                print(f'Round {round_idx + 1}, Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    config = Config()
    fedbkt = FedBKT(config)
    fedbkt.train() 