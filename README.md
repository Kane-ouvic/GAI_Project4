markdown

# Generative Models for Visual Signals – Assignment

This repository contains the implementation of the assignment for integrating DDPM and DIP models for image generation and restoration tasks using the CIFAR-10 dataset.

## Setup

### Clone the repository:
```
   git clone https://github.com/your_username/GenAI_Assignment.git
   cd GenAI_Assignment

    Install the required packages:

    bash

    pip install -r requirements.txt
```
##  Usage

### Train DIP Model

To train the DIP model on the CIFAR-10 dataset:

```
python train_dip.py --output_path path/to/save/dip_model.pth --data_path path/to/cifar-10-batches-py
```

### Train DDPM Model with DIP Prior

To train the DDPM model using the DIP prior:

```
python train_ddpm.py --dip_model_path path/to/save/dip_model.pth --output_path path/to/save/ddpm_model.pth --data_path path/to/cifar-10-batches-py
```

### Evaluate Models

To evaluate the trained models:

```
python evaluate.py --ddpm_model_path path/to/save/ddpm_model.pth --data_path path/to/cifar-10-batches-py
```

### Directory Structure

    models/: Contains the implementations of DIP and DDPM models.
    utils/: Contains utility functions for data loading and preprocessing.