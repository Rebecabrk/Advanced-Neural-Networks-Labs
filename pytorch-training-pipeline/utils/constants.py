from pathlib import Path
import yaml

DEFAULT_CONFIG_PATH = Path("/Users/rebecacostache/Desktop/Advanced-Neural-Networks-Labs/pytorch-training-pipeline/configs/default_config.yaml")

VALID_DATASETS = ["MNIST", "CIFAR-10", "CIFAR-100", "OxfordIIITPet"]
VALID_MODELS = ["resnet18", "resnet50", "resnest14d", "resnest26d", "MLP"]
VALID_OPTIMIZERS = ["SGD", "Adam", "AdamW", "Muon", "SAM"]
VALID_SCHEDULERS = ["StepLR", "ReduceLROnPlateau"]

with open("configs/optimizer_params.yaml", "r") as f:
    OPTIMIZER_PARAMS = yaml.safe_load(f)