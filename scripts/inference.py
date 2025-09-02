#!/usr/bin/env python3
"""
Inference script for MMFusion-IML models
"""

import argparse
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='Run inference with MMFusion-IML models')
    parser.add_argument('--exp', type=str, help='Path to experiment config file')
    parser.add_argument('--ckpt', type=str, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to output directory')
    
    args = parser.parse_args()
    
    from inference import main as inference_main
    
    # Modify sys.argv to match expected format for inference script
    sys.argv = ['inference.py']
    if args.exp:
        sys.argv.extend(['--exp', args.exp])
    if args.ckpt:
        sys.argv.extend(['--ckpt', args.ckpt])
    if args.input:
        sys.argv.extend(['--input', args.input])
    if args.output:
        sys.argv.extend(['--output', args.output])
        
    inference_main()

if __name__ == '__main__':
    main()
