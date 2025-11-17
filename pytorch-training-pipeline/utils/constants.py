from pathlib import Path
import yaml
import os

# assuming that the script is always run from 
# /Advanced-Neural-Networks-Labs/pytorch-training-pipeline
ROOT = os.getcwd() 
DEFAULT_CONFIG_PATH = Path(ROOT + "/configs/default_config.yaml")

VALID_DATASETS = ["MNIST", "CIFAR-10", "CIFAR-100", "OxfordIIITPet"]
VALID_MODELS = ["resnet18", "resnet50", "resnest14d", "resnest26d", "MLP"]
VALID_OPTIMIZERS = ["SGD", "Adam", "AdamW", "Muon", "SAM"]
VALID_SCHEDULERS = ["StepLR", "ReduceLROnPlateau"]

with open(
    ROOT  + "/configs/optimizers.yaml", 
    "r") as f:
    OPTIMIZER_PARAMS = yaml.safe_load(f)

with open(
    ROOT + "/configs/transforms.yaml",
      "r") as f:
    TRANSFORM_PRESETS = yaml.safe_load(f)

with open(
    ROOT + "/configs/MLP.yaml",
      "r") as f:
    MLP_PARAMS = yaml.safe_load(f)
    
with open(
    ROOT + "/configs/schedulers.yaml",
      "r") as f:
    SCHEDULER_PARAMS = yaml.safe_load(f)
