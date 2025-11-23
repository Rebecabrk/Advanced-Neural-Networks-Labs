# PyTorch Training Pipeline Usage Guide

This guide explains how to use the provided PyTorch training pipeline.

## Features
- Flexible model selection (MLP, timm models)
- Data augmentation and transform customization
- Optimizer selection (SGD, Adam, AdamW, SAM, Muon*)
- Learning rate scheduler support (StepLR, ReduceLROnPlateau)
- Batch size scheduling
- Early stopping
- TensorBoard and wandb logging
- Colab and local compatibility (see https://colab.research.google.com/drive/1bgXBqE6aZuBV1N9VSpLUbZira5aE0XJ5?usp=sharing)

*Note: Muon only works with MLPs and no bias terms. Because Muon is an optimizer for 2D parameters, I needed to enforce the biases (which be default are 1D) to be False, such that only the weights remain (weights are 2D tensors).

---

## 1. Configuration

All settings are controlled via YAML config files (e.g., `configs/default_config.yaml`).

### Example config:
```yaml
model:
  name: "MLP"
  mlp_params:
    hidden_layers: [1024, 512, 256]
    activation: ["ReLU", "ReLU", "Sigmoid"]
    dropout: 0.5
optimizer:
  name: "SGD"
  lr: 0.001
  weight_decay: 0.01
  sgd_params:
    momentum: 0.9
    dampening: 0.0
    nesterov: False
training:
  epochs: 10
  batch_size: 128
  batch_scheduler:
    enabled: True
    schedule_epochs: [2, 5, 8]
    batch_size_increments: [128, 256, 512]
lr_scheduler:
  enabled: True
  name: "StepLR"
  step_size: 15
  gamma: 0.1
```

---


## 2. Running the Pipeline

You can run the script **with or without a config file**:

- **With a config file:**
  1. Edit your config in `configs/default_config.yaml` or create a new one.
  2. Run:
     ```bash
     python pytorch-training-pipeline/main.py --config configs/default_config.yaml
     ```

- **Without a config file:**
  - Simply run:
    ```bash
    python pytorch-training-pipeline/main.py
    ```
  - The script will use built-in defaults or interactively prompt you for configuration options.

The pipeline asks the user if they want to save the customized config. If yes, it will be saved as a yaml file.
> The script must be run from the root project directory (/Advanced-Neural-Networks-Labs)

**Monitor logs:**
- TensorBoard: Open a terminal and run `tensorboard --logdir=runs`, then open the provided URL in your browser.
- wandb: Check your project dashboard.

I do not have any photos of the graphs of wandb as my free trial ended and I am not allowed to see my runs anymore. If this is a problem, I kindly ask you to write to me. I will look into it i try to access the 

---

## 3. Available Datasets
The pipeline supports the following datasets (set in your config under `data.dataset_name`):

- `CIFAR-10`: 10-class color image classification (32x32)
- `CIFAR-100`: 100-class color image classification (32x32)
- `MNIST`: 10-class grayscale digit classification (28x28)
- `OxfordIIITPet`: 37-class pet breed classification (images are resized to 224x224 by default)

---

## 4. Supported Models
- `MLP`: Fully connected, customizable layers/activations
- The following models using timm: `resnet18`, `resnet50`, `resnest14d`, `resnest26d`
- The above models can be pretrained or not.

---

## 5. Data Augmentation & Available Transforms
You can enable, disable, and customize data augmentation transforms in your config under `data.transforms`.
The following transforms are available (see `/configs/transforms.yaml`):

- **random_horizontal_flip**
- **random_vertical_flip**
- **random_rotation**
- **random_crop**
- **color_jitter**
- **random_grayscale**
- **random_erasing**
- **normalization**

Each transform can be enabled/disabled and has parameters (like probability, size, degrees, etc.) that you can customize in your config.

---

## 6. Optimizers
The pipeline supports the following optimizers, each with customizable parameters (see `/configs/optimizers.yaml`):

### SGD
- **lr**: Learning rate (float, e.g., 0.001)
- **weight_decay**: Weight decay (float)
- **momentum**: Momentum factor (float)
- **dampening**: Dampening for momentum (float)
- **nesterov**: Enables Nesterov momentum (boolean)

### Adam / AdamW
- **lr**: Learning rate (float)
- **weight_decay**: Weight decay (float)
- **betas**: Coefficients for computing running averages of gradient and its square (list of two floats, e.g., [0.9, 0.999])
- **eps**: Term added to denominator to improve numerical stability (float)
- **amsgrad**: Use the AMSGrad variant (boolean)

### Muon
- **lr**: Learning rate (float)
- **weight_decay**: Weight decay (float)
- **momentum**: Momentum factor (float)
- **ns_steps**: Number of negative curvature steps (int)

### SAM
- **lr**: Learning rate (float)
- **weight_decay**: Weight decay (float)
- **base_optimizer**: Which optimizer to use as the base (SGD, Adam, AdamW)
- **rho**: Neighborhood size (float)
- **sgd_params**: (if base_optimizer is SGD) momentum, dampening, nesterov

**Note:**
- For all optimizers, `lr` and `weight_decay` are required.
- For Muon, only use with MLPs and set all `bias=False`.
- For SAM, set `base_optimizer` and its parameters in `sam_params`.

---

## 7. Learning Rate Schedulers
The pipeline supports the following learning rate schedulers, which you can configure under `lr_scheduler` in your config. You need to specify the following params for each scheduler:

### StepLR
- **step_size**: Number of epochs between each learning rate decay (integer, e.g., 10).
- **gamma**: Multiplicative factor of learning rate decay (float, e.g., 0.1).

### ReduceLROnPlateau
- **mode**: One of `min` or `max`. In `min` mode, learning rate will be reduced when the monitored quantity has stopped decreasing; in `max` mode it will be reduced when the quantity has stopped increasing.
- **factor**: Factor by which the learning rate will be reduced (float, e.g., 0.1).
- **patience**: Number of epochs with no improvement after which learning rate will be reduced (integer).
- **threshold**: Threshold for measuring the new optimum, to only focus on significant changes (float).
- **threshold_mode**: One of `relative` or `absolute`. In `relative` mode, dynamic threshold is relative to best; in `absolute` mode, it is absolute.

Refer to `/configs/schedulers.yaml` for default values and valid ranges for each parameter.

---

## 8. Batch Size Scheduler 
You can dynamically change the training batch size during training by enabling the batch size scheduler in your config under `training.batch_scheduler`.

### Parameters:
- **enabled**: Set to `True` to activate the batch size scheduler.
- **schedule_epochs**: List of epochs at which to change the batch size (e.g., `[2, 5, 8]`).
- **batch_size_increments**: List of batch sizes to use at the corresponding epochs (e.g., `[128, 256, 512]`).

At each epoch listed in `schedule_epochs`, the batch size will be updated to the corresponding value in `batch_size_increments`.

---

## 9. Early Stopping
The pipeline also supports an early stopping mechanism. You can enable and configure early stopping in your config under `training.early_stopping`.

### Parameters:
- **enabled**: Set to `True` to activate early stopping.
- **patience**: Number of epochs with no improvement in validation loss before stopping training early (integer, e.g., 5).

When enabled, training will stop if the validation loss does not improve for the specified number of epochs. A message will be printed in the terminal when early stopping is triggered.

---

## 10. Estimation of grade
After evaluating my homework, I believe I will receive 8 points (for the first batch of instructions) and 1 bonus point for integrating the code on google colab to run using GPU T4 and for the fact that the MLP is highly customizable (the number of layers, the size of each layer, each activation funciton and dropout).