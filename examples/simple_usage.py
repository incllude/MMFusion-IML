#!/usr/bin/env python3
"""
Simple usage example for MMFusion-IML library

This example shows how to:
1. Load and use models
2. Create datasets
3. Run basic inference
"""

import torch
import numpy as np
import cv2
import sys
import os

# Add parent directory to path for importing mmfusion_iml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import mmfusion_iml as mmf
    print("✓ Successfully imported mmfusion_iml")
except ImportError as e:
    print(f"✗ Failed to import mmfusion_iml: {e}")
    print("Make sure to install the package first: pip install -e .")
    sys.exit(1)

def example_model_usage():
    """Example of loading and using models"""
    print("\n=== Model Usage Example ===")
    
    try:
        # Create a simple config object
        class SimpleConfig:
            BACKBONE = 'CMNeXtMHSA-B2'
            NUM_CLASSES = 2
            MODALS = ['rgb']
            TRAIN_PHASE = 'localization'
            DETECTION = 'confpool'
            PRETRAINED = None  # No pretrained weights for this example
        
        config = SimpleConfig()
        
        # Initialize model
        print("Creating CMNeXtConf model...")
        model = mmf.models.CMNeXtConf(config)
        model.eval()
        print(f"✓ Model created successfully: {type(model).__name__}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 512, 512)
        print(f"✓ Created dummy input: {dummy_input.shape}")
        
        # Run forward pass
        with torch.no_grad():
            output = model([dummy_input])
            print(f"✓ Forward pass successful. Output shape: {output.shape}")
            
    except Exception as e:
        print(f"✗ Model usage failed: {e}")

def example_utilities():
    """Example of using utility functions"""
    print("\n=== Utilities Example ===")
    
    try:
        # SRM Filter
        print("Testing SRM Filter...")
        srm_filter = mmf.common.SRMFilter()
        dummy_image = torch.randn(1, 3, 256, 256)
        srm_output = srm_filter(dummy_image)
        print(f"✓ SRM Filter output shape: {srm_output.shape}")
        
        # Bayar Convolution
        print("Testing Bayar Convolution...")
        bayar_conv = mmf.common.BayarConv2d(3, 3, padding=2)
        bayar_output = bayar_conv(dummy_image)
        print(f"✓ Bayar Conv output shape: {bayar_output.shape}")
        
        # Average Meter
        print("Testing Average Meter...")
        meter = mmf.common.AverageMeter()
        meter.update(0.5)
        meter.update(0.7)
        print(f"✓ Average Meter value: {meter.average():.3f}")
        
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")

def example_modal_extractor():
    """Example of using modal extractor"""
    print("\n=== Modal Extractor Example ===")
    
    try:
        # Create modal extractor (without noiseprint weights)
        print("Creating Modal Extractor...")
        modal_extractor = mmf.models.ModalExtract(
            modals=['bayar', 'srm']  # Skip noiseprint as it requires weights
        )
        modal_extractor.eval()
        
        # Test with dummy image
        dummy_image = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            modalities = modal_extractor(dummy_image)
            print(f"✓ Modal Extractor created {len(modalities)} modalities")
            for i, modal in enumerate(modalities):
                print(f"  Modal {i}: {modal.shape}")
                
    except Exception as e:
        print(f"✗ Modal Extractor test failed: {e}")

def example_dataset_creation():
    """Example of creating datasets (without actual data)"""
    print("\n=== Dataset Creation Example ===")
    
    try:
        # Note: This will fail without actual data files, but shows the API
        print("Dataset API demonstration (will fail without data files):")
        print("  - ManipulationDataset(path='data.txt', image_size=512, train=True)")
        print("  - MixDataset(paths=['data1.txt', 'data2.txt'], image_size=512)")
        print("✓ Dataset API imported successfully")
        
        # Test dataset utilities
        print("Testing dataset utilities...")
        from mmfusion_iml.data.datasets import get_random_crop_coords_on_grid
        coords = get_random_crop_coords_on_grid(1024, 1024, 512, 512, 0.5, 0.5)
        print(f"✓ Random crop coordinates: {coords}")
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")

def main():
    """Run all examples"""
    print("MMFusion-IML Library Usage Examples")
    print("=" * 40)
    
    # Test basic imports
    print("\n=== Import Test ===")
    print(f"✓ mmfusion_iml version: {mmf.__version__}")
    print(f"✓ Available modules: {[name for name in dir(mmf) if not name.startswith('_')]}")
    
    # Run examples
    example_utilities()
    example_modal_extractor()
    example_model_usage()
    example_dataset_creation()
    
    print("\n" + "=" * 40)
    print("Examples completed!")
    print("\nNext steps:")
    print("1. Install pretrained weights (see pretrained/README.md)")
    print("2. Prepare your datasets (see data/ directory)")
    print("3. Create configuration files (see experiments/ directory)")
    print("4. Check USAGE.md for detailed documentation")

if __name__ == "__main__":
    main()
