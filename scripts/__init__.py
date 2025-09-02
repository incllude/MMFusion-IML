"""
Scripts module for MMFusion-IML

This module contains command-line scripts for training, testing, and inference.
"""

from . import train
from . import test  
from . import inference

__all__ = [
    "train",
    "test", 
    "inference",
]
