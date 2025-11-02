# 🚨 EMERGENCY FIX - Run This NOW in Kaggle

## ❌ Error:
```
ImportError: cannot import name 'preserve_channel_dim' from 'albucore.utils'
```

## ✅ IMMEDIATE SOLUTION - Copy and Run This Cell:

**Add a NEW cell right after your dependency installation and run this:**

```python
# 🔧 EMERGENCY FIX for albumentations/albucore compatibility
print("🚨 Fixing albumentations/albucore ImportError...")

# Force uninstall both packages
!pip uninstall -y albumentations albucore -q

# Reinstall with matching versions
!pip install -q albucore==0.0.24
!pip install -q albumentations==2.0.8

# Verify the fix
print("\n🔍 Verifying fix...")
try:
    import albumentations as A
    import albucore
    from albucore.utils import preserve_channel_dim
    
    print(f"✅ Albumentations: {A.__version__}")
    print(f"✅ Albucore: {albucore.__version__}")
    print("✅ ImportError FIXED - 'preserve_channel_dim' imported successfully!")
    print("\n🚀 You can now continue with training!")
except ImportError as e:
    print(f"❌ Still failing: {e}")
    print("⚠️  Try: Runtime → Restart Runtime, then re-run ALL cells")
```

---

## 📋 Alternative: Update Your Dependency Cell

**OR replace your entire dependency installation cell (Step 2) with:**

```python
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

# ⚠️ CRITICAL FIX: Uninstall existing albumentations/albucore first
print("🔄 Fixing albumentations/albucore compatibility...")
!pip uninstall -y albumentations albucore -q
!pip install -q albucore==0.0.24
!pip install -q albumentations==2.0.8

# Install remaining packages
!pip install -q timm==0.9.16 scikit-learn==1.3.2

print("\n✅ All dependencies installed!")
```

---

## 🔄 If Still Failing:

1. **Restart the Kernel**: Runtime → Restart Runtime
2. **Re-run ALL cells** from the beginning
3. The uninstall/reinstall should work on fresh kernel

---

## Why This Happens:

Kaggle has `albumentations 2.0.8` pre-installed, but when other packages install, they may bring in an incompatible `albucore` version. The **explicit uninstall and reinstall** forces the correct versions.

---

## 🚀 After the Fix:

Once you see:
```
✅ ImportError FIXED - 'preserve_channel_dim' imported successfully!
```

**Continue with the rest of your notebook** - training will work!

---

**COPY THE CODE ABOVE AND RUN IT NOW!** 🚨
