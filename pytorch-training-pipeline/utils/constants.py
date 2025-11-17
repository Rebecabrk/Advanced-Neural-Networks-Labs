from pathlib import Path
import yaml

DEFAULT_CONFIG_PATH = Path("/content/Advanced-Neural-Networks-Labs/pytorch-training-pipeline/configs/default_config.yaml")

VALID_DATASETS = ["MNIST", "CIFAR-10", "CIFAR-100", "OxfordIIITPet"]
VALID_MODELS = ["resnet18", "resnet50", "resnest14d", "resnest26d", "MLP"]
VALID_OPTIMIZERS = ["SGD", "Adam", "AdamW", "Muon", "SAM"]
VALID_SCHEDULERS = ["StepLR", "ReduceLROnPlateau"]

with open(
    "/content/Advanced-Neural-Networks-Labs/pytorch-training-pipeline/configs/optimizers.yaml", 
    "r") as f:
    OPTIMIZER_PARAMS = yaml.safe_load(f)

with open(
    "/content/Advanced-Neural-Networks-Labs/pytorch-training-pipeline/configs/transforms.yaml",
      "r") as f:
    TRANSFORM_PRESETS = yaml.safe_load(f)

with open(
    "/content/Advanced-Neural-Networks-Labs/pytorch-training-pipeline/configs/MLP.yaml",
      "r") as f:
    MLP_PARAMS = yaml.safe_load(f)
    
with open(
    "/content/Advanced-Neural-Networks-Labs/pytorch-training-pipeline/configs/schedulers.yaml",
      "r") as f:
    SCHEDULER_PARAMS = yaml.safe_load(f)
