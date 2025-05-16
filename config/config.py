class Config:
    # 数据集配置
    dataset = 'cifar10'  # 可选: 'cifar10', 'cifar100', 'fashion_mnist', 'flowers102'
    num_clients = 20
    batch_size = 100
    num_epochs = 100
    num_rounds = 1000
    
    # 模型配置
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    
    # 知识迁移配置
    temperature = 2.0  # 知识蒸馏温度
    alpha = 0.5  # 知识迁移权重
    
    # 非独立同分布配置
    alpha_dirichlet = 0.1  # Dirichlet分布参数，控制数据分布的不平衡程度
    
    # 设备配置
    device = 'cuda'  # 或 'cpu'
    
    # 路径配置
    data_dir = './data'
    model_dir = './models'
    log_dir = './logs' 