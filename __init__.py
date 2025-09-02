"""
MMFusion-IML: Multi-Modal Fusion for Image Manipulation Detection and Localization

This package provides state-of-the-art models and utilities for detecting and localizing
image manipulations using multi-modal fusion techniques.

Main components:
- models: Deep learning models for manipulation detection/localization
- data: Dataset utilities and data loaders
- common: Shared utilities, losses, metrics
- configs: Model configurations
"""

__version__ = "0.1.0"
__author__ = "Konstantinos Triaridis, Vasileios Mezaris"

# Import main components for easy access
from . import models
from . import data
from . import common
from . import configs

# Import key classes and functions
from .models.base import BaseModel
from .models.cmnext_conf import CMNeXtConf
from .models.ws_cmnext_conf import WSCMNeXtConf
from .data.datasets import ManipulationDataset, MixDataset
from .common.utils import AverageMeter, SRMFilter, BayarConv2d

__all__ = [
    # Main modules
    "models",
    "data", 
    "common",
    "configs",
    # Key classes
    "BaseModel",
    "CMNeXtConf", 
    "WSCMNeXtConf",
    "ManipulationDataset",
    "MixDataset",
    "AverageMeter",
    "SRMFilter", 
    "BayarConv2d",
]
