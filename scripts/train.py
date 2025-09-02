#!/usr/bin/env python3
"""
Training script for MMFusion-IML models
"""

import argparse
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='Train MMFusion-IML models')
    parser.add_argument('--exp', type=str, required=True, help='Path to experiment config file')
    parser.add_argument('--ckpt', type=str, help='Path to checkpoint (for resuming training)')
    parser.add_argument('--phase', type=str, choices=['localization', 'detection'], 
                       default='localization', help='Training phase')
    
    args = parser.parse_args()
    
    if args.phase == 'localization':
        from ec_train import main as train_main
    else:
        from ec_train_phase2 import main as train_main
        
    # Modify sys.argv to match expected format for training scripts
    sys.argv = ['train.py', '--exp', args.exp]
    if args.ckpt:
        sys.argv.extend(['--ckpt', args.ckpt])
        
    train_main()

if __name__ == '__main__':
    main()
