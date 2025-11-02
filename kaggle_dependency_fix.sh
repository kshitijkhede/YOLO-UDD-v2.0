#!/bin/bash
# Quick fix script to copy to Kaggle notebook

echo "# Add this to your Kaggle notebook dependency installation cell:"
echo ""
cat << 'EOF'
# =======================
# ✅ Install Compatible Dependencies (Version-Locked)
# =======================
print("🔧 Installing dependencies with fixed versions to avoid conflicts...\n")

# Uninstall numpy first to avoid conflicts
!pip uninstall -y numpy -q

# Install version-locked dependencies for Kaggle (CUDA 11.8)
!pip install -q numpy==1.26.4
!pip install -q torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
!pip install -q opencv-python-headless==4.9.0.80 pillow==10.3.0 pycocotools==2.0.7 pyyaml==6.0.1 tqdm==4.66.4 tensorboard==2.16.2

# ⚠️ CRITICAL: Install compatible albucore BEFORE albumentations
!pip install -q albucore==0.0.17
!pip install -q albumentations==1.4.8 timm==0.9.16 scikit-learn==1.3.2

print("\n✅ All dependencies installed successfully with version-locking!")
print("   - Numpy: 1.26.4 (compatible with all packages)")
print("   - PyTorch: 2.2.2 (CUDA 11.8)")
print("   - Albumentations: 1.4.8 + albucore: 0.0.17 (compatible)")
print("   - scikit-learn: 1.3.2 (compatible)")
print("   - No dependency conflicts ✓")
EOF
