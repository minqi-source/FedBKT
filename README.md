# FedBKT: Federated Learning with Bidirectional Knowledge Transfer

This repository contains the implementation of FedBKT, a novel federated learning framework that enables efficient knowledge transfer between heterogeneous models while preserving model personalization.

## Paper Information

**Title**: FedBKT: Federated Learning with Bidirectional Knowledge Transfer

**Authors**: Qi Min, Wei Chen, et al.

**Conference**: ICIC 2025 (International Conference on Intelligent Computing)

## Framework Overview

![FedBKT Framework](figures/frame.pdf)

The framework consists of three main components:
1. **Mediator Model**: A central model that facilitates knowledge exchange between clients
2. **Knowledge Sharing Module**: Transfers global knowledge from the mediator to local models
3. **Knowledge Extraction Module**: Extracts local knowledge to update the global model

## Key Features

- Bidirectional knowledge transfer between clients and server
- No reliance on public datasets
- Hierarchical distillation strategy
- Adaptive weight mechanism based on forgetting degree
- Support for heterogeneous model architectures
- Efficient handling of non-IID data distributions

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```python
from config.config import Config
from train import FedBKT

# Initialize configuration
config = Config()

# Create and train the model
fedbkt = FedBKT(config)
fedbkt.train()
```

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{fedbkt2025,
  title={FedBKT: Federated Learning with Bidirectional Knowledge Transfer},
  author={Min, Qi and Chen, Wei and others},
  booktitle={International Conference on Intelligent Computing (ICIC)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We thank all the contributors and reviewers for their valuable feedback and suggestions. 