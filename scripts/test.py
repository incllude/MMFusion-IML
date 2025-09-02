#!/usr/bin/env python3
"""
Testing script for MMFusion-IML models
"""

import argparse
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='Test MMFusion-IML models')
    parser.add_argument('--exp', type=str, required=True, help='Path to experiment config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--task', type=str, choices=['detection', 'localization'], 
                       default='localization', help='Task type')
    parser.add_argument('--manip', type=str, help='Path to manipulated images list')
    parser.add_argument('--auth', type=str, help='Path to authentic images list (for detection)')
    
    args = parser.parse_args()
    
    if args.task == 'localization':
        from test_localization import main as test_main
        sys.argv = ['test_localization.py', '--exp', args.exp, '--ckpt', args.ckpt]
        if args.manip:
            sys.argv.extend(['--manip', args.manip])
    else:
        from test_detection import main as test_main
        sys.argv = ['test_detection.py', '--exp', args.exp, '--ckpt', args.ckpt]
        if args.manip:
            sys.argv.extend(['--manip', args.manip])
        if args.auth:
            sys.argv.extend(['--auth', args.auth])
        
    test_main()

if __name__ == '__main__':
    main()
