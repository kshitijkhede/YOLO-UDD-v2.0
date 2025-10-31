"""
Quick test to verify training setup works correctly
"""
import torch
import sys
import os

print("=" * 80)
print("YOLO-UDD v2.0 - Training Setup Test")
print("=" * 80)

# 1. Check CUDA availability
print("\n1. Checking CUDA availability...")
if torch.cuda.is_available():
    print(f"   ✅ CUDA is available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print(f"   ⚠️  CUDA not available, will use CPU")

# 2. Test model import
print("\n2. Testing model import...")
try:
    from models import build_yolo_udd
    print(f"   ✅ Model imported successfully")
except Exception as e:
    print(f"   ❌ Error importing model: {e}")
    sys.exit(1)

# 3. Test dataset loading
print("\n3. Testing dataset loading...")
try:
    from data.dataset import TrashCanDataset, create_dataloaders
    
    dataset = TrashCanDataset(
        data_dir='data/trashcan',
        split='train',
        img_size=640,
        augment=False
    )
    print(f"   ✅ Dataset loaded successfully")
    print(f"   Number of samples: {len(dataset)}")
    
    # Test loading one sample
    sample = dataset[0]
    print(f"   ✅ Sample loaded: image shape {sample['image'].shape}")
    
except Exception as e:
    print(f"   ❌ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test model creation
print("\n4. Testing model creation...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_yolo_udd(num_classes=22, pretrained=None)
    model = model.to(device)
    print(f"   ✅ Model created successfully")
    print(f"   Device: {device}")
    
    # Get model info
    model_info = model.get_model_info()
    total_params = model_info['Total Parameters']
    print(f"   Total parameters: {total_params}")
    
except Exception as e:
    print(f"   ❌ Error creating model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Test forward pass
print("\n5. Testing forward pass...")
try:
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        predictions, turb_score = model(dummy_input)
    print(f"   ✅ Forward pass successful")
    print(f"   Number of detection scales: {len(predictions)}")
    print(f"   Turbidity score shape: {turb_score.shape}")
    
except Exception as e:
    print(f"   ❌ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Test dataloader
print("\n6. Testing dataloader...")
try:
    dataloaders = create_dataloaders(
        data_dir='data/trashcan',
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        img_size=640
    )
    
    # Get one batch
    batch = next(iter(dataloaders['train']))
    print(f"   ✅ Dataloader works")
    print(f"   Batch size: {batch['images'].shape[0]}")
    print(f"   Image shape: {batch['images'].shape}")
    
except Exception as e:
    print(f"   ❌ Error in dataloader: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. Test loss function
print("\n7. Testing loss function...")
try:
    from utils.loss import YOLOUDDLoss
    
    criterion = YOLOUDDLoss(num_classes=22)
    print(f"   ✅ Loss function created successfully")
    
except Exception as e:
    print(f"   ❌ Error creating loss function: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 8. Check config file
print("\n8. Checking configuration file...")
try:
    import yaml
    
    config_path = 'configs/train_config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✅ Config file loaded")
        print(f"   Batch size: {config['training']['batch_size']}")
        print(f"   Epochs: {config['training']['epochs']}")
        print(f"   Learning rate: {config['training']['learning_rate']}")
    else:
        print(f"   ⚠️  Config file not found at {config_path}")
        
except Exception as e:
    print(f"   ❌ Error loading config: {e}")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED! Your training setup is ready!")
print("=" * 80)
print("\nTo start training, run:")
print("  python scripts/train.py --config configs/train_config.yaml")
print("\nFor quick test (2 epochs):")
print("  python scripts/train.py --config configs/train_config.yaml --epochs 2 --batch-size 4")
print("=" * 80)
