import yaml
from enum import Enum
import os

SRC_ROOT = os.path.split(os.path.abspath(__file__))[0]
SOFTWARE_CONFIG_FILE = os.path.join(SRC_ROOT, "software.yaml")
CONFIG_FILE = os.path.join(SRC_ROOT, "config.yaml")
CHECKPOINT_PATH = os.path.join(SRC_ROOT, "checkpoints")

from typing import Dict


def load_config(config_file: str) -> Dict:
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_config(config, config_file: str) -> bool:
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return True
