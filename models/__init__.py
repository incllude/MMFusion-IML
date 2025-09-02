"""
Models module for MMFusion-IML

This module contains deep learning models for image manipulation detection and localization.
"""

from .base import BaseModel, load_dualpath_model
from .cmnext_conf import CMNeXtConf
from .ws_cmnext_conf import WSCMNeXtConf
from .DnCNN_noiseprint import DnCNN, DnCNNNoiseprint
from .modal_extract import ModalExtract, ModalitiesExtractor

# Import submodules
from . import backbones
from . import heads
from . import layers
from . import modules

__all__ = [
    # Main model classes
    "BaseModel",
    "CMNeXtConf", 
    "WSCMNeXtConf",
    "DnCNN",
    "DnCNNNoiseprint",
    "ModalExtract",
    "ModalitiesExtractor",
    # Utility functions
    "load_dualpath_model",
    # Submodules
    "backbones",
    "heads", 
    "layers",
    "modules",
]
