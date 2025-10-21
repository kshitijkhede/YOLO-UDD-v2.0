# Running YOLO-UDD v2.0 on Google Colab

## Quick Start with Google Colab

### Step 1: Open the Notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Open notebook** → **GitHub** tab
3. Enter: `https://github.com/kshitijkhede/YOLO-UDD-v2.0`
4. Select `YOLO_UDD_Training.ipynb`

### Step 2: Enable GPU
1. Click **Runtime** → **Change runtime type**
2. Select **GPU** (T4 or better)
3. Click **Save**

### Step 3: Run All Cells
- Click **Runtime** → **Run all**
- Or run cells sequentially with `Shift + Enter`

## Benefits of Using Colab

✅ **Free GPU Access**: T4 GPU (16GB VRAM) for free  
✅ **Faster Training**: ~100x faster than CPU training  
✅ **No Setup Required**: All dependencies installed automatically  
✅ **Easy Sharing**: Share notebook with collaborators  
✅ **Cloud Storage**: Save results to Google Drive  

## Estimated Training Times

| Device | Batch Size | Time per Epoch | 50 Epochs |
|--------|------------|----------------|-----------|
| CPU (Local) | 4 | ~30 seconds | ~25 minutes |
| GPU T4 (Colab) | 16 | ~2 seconds | ~2 minutes |
| GPU V100 (Colab Pro) | 32 | ~1 second | ~1 minute |

## Dataset Options

### Option A: Use Dummy Data (Testing)
- Already configured in the notebook
- Good for testing the pipeline
- No dataset upload needed

### Option B: Upload Your Dataset to Google Drive
1. Upload TrashCAN dataset to Google Drive
2. Mount Drive in Colab (provided in notebook)
3. Link to dataset directory

```python
# In Colab notebook
from google.colab import drive
drive.mount('/content/drive')

# Link to your dataset
!ln -s /content/drive/MyDrive/TrashCAN_Dataset data/trashcan
```

## Download Trained Model

After training completes, download the best model:

```python
from google.colab import files
files.download('runs/train/checkpoints/best.pt')
```

Or save to Google Drive:

```python
!cp -r runs/train/* /content/drive/MyDrive/YOLO-UDD-Results/
```

## Monitoring Training

### TensorBoard in Colab
```python
%load_ext tensorboard
%tensorboard --logdir runs/train/logs
```

### View Metrics
- Loss curves (bbox, objectness, classification)
- Turbidity scores
- mAP metrics
- Learning rate schedule

## Troubleshooting

### Out of Memory Error
Reduce batch size:
```python
BATCH_SIZE = 8  # Or 4 if still having issues
```

### Session Timeout
Colab free tier disconnects after ~12 hours. To prevent:
1. Upgrade to Colab Pro ($9.99/month)
2. Or run shorter training sessions (25 epochs each)
3. Save checkpoints regularly (done automatically)

### Slow Upload
If uploading large dataset is slow:
1. Use `gdown` to download from Google Drive link
2. Or use dummy data for pipeline testing

## Alternative: Kaggle Notebooks

You can also use Kaggle Notebooks (similar to Colab):
1. Create account at [kaggle.com](https://www.kaggle.com)
2. Create new notebook
3. Enable GPU accelerator
4. Clone repository and run training

## Local Training (Comparison)

If you prefer local training:

```bash
cd /home/student/MIR/Project/YOLO-UDD-v2.0
./run_training.sh
```

But note: **CPU training is ~100x slower** than Colab GPU!

## Next Steps

After training on Colab:
1. Download `best.pt` model
2. Run inference locally or on Colab
3. Evaluate on test set
4. Deploy for real-world use

## Questions?

Check the main [README.md](README.md) for more details or open an issue on GitHub.
