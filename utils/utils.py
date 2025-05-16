import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import torch.nn.functional as F

def get_dataset(dataset_name, data_dir):
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                              download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                             download=True, transform=transform_test)
        num_classes = 10
    # 可以添加其他数据集的加载逻辑
    return trainset, testset, num_classes

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idxs = [np.where(train_labels == i)[0] for i in range(n_classes)]
    client_idxs = [[] for _ in range(n_clients)]
    
    for c, fracs in zip(class_idxs, label_distribution):
        for i, idxs in enumerate(np.split(c, (np.cumsum(fracs) * len(c)).astype(int))):
            client_idxs[i].extend(idxs)
    
    return client_idxs

def create_client_dataloaders(trainset, n_clients, alpha, batch_size):
    train_labels = np.array(trainset.targets)
    client_idxs = dirichlet_split_noniid(train_labels, alpha, n_clients)
    
    client_loaders = []
    for idxs in client_idxs:
        client_dataset = Subset(trainset, idxs)
        client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        client_loaders.append(client_loader)
    
    return client_loaders

def kl_divergence(p, q, temperature=1.0):
    p = F.softmax(p / temperature, dim=1)
    q = F.softmax(q / temperature, dim=1)
    return F.kl_div(p.log(), q, reduction='batchmean') * (temperature ** 2)

def calculate_forgetting_degree(model, old_data, new_data, device):
    model.eval()
    with torch.no_grad():
        old_outputs = model(old_data.to(device))
        new_outputs = model(new_data.to(device))
        forgetting = F.kl_div(
            F.log_softmax(old_outputs, dim=1),
            F.softmax(new_outputs, dim=1),
            reduction='batchmean'
        )
    return forgetting.item() 