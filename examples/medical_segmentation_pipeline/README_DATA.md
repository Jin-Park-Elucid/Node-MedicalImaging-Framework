# Dataset Information

## ⚠️ IMPORTANT: This is Synthetic Data

The data in this example is **completely synthetic** - it is NOT real medical imaging data.

### What You're Training On

**Synthetic images created by `download_dataset.py`**:
- Random Gaussian noise (gray background)
- Random circular bright regions (simulating "lesions")
- Binary masks marking where circles are

**This is designed for**:
- ✅ Testing the framework
- ✅ Learning the workflow
- ✅ Demonstrating the code works
- ✅ Quick prototyping

**This is NOT**:
- ❌ Real CT scans
- ❌ Real MRI images
- ❌ Actual anatomical structures
- ❌ Clinically useful

---

## Dataset Structure

```
data/
├── dataset_info.json     # Dataset metadata
├── train/                # Training data (50 images)
│   ├── images/
│   │   ├── image_0000.png
│   │   ├── image_0001.png
│   │   └── ...
│   └── masks/
│       ├── mask_0000.png
│       ├── mask_0001.png
│       └── ...
└── test/                 # Test data (20 images)
    ├── images/
    │   ├── image_0000.png
    │   ├── image_0001.png
    │   └── ...
    └── masks/
        ├── mask_0000.png
        ├── mask_0001.png
        └── ...
```

---

## Visualize the Data

To see what the raw data looks like:

```bash
cd examples/medical_segmentation_pipeline
python inspect_dataset.py
```

This will show you:
- 6 random training samples
- Images vs masks side-by-side
- Data statistics
- Save visualization to `dataset_visualization.png`

---

## Why Metrics Are So High

When you train and test on this data:

**Expected results**:
- Dice Score: ~0.95-0.97
- Accuracy: ~0.99
- IoU: ~0.90-0.95

**Why so good**:
- Task is trivial: find bright circles in noise
- Model easily learns this pattern
- 50 training samples is plenty
- **NOT representative of real medical imaging difficulty!**

**Real medical data metrics**:
- Brain tumor segmentation: 0.75-0.85 Dice
- Lung lesion detection: 0.70-0.80 Dice
- Multi-organ segmentation: 0.70-0.90 Dice

---

## Using Real Medical Data

### Option 1: Medical Segmentation Decathlon

1. **Download dataset**:
   ```bash
   # Visit: http://medicaldecathlon.com/
   # Download Task01_BrainTumour (or other task)
   ```

2. **Convert to PNG** (if needed):
   ```python
   # Most datasets come as NIfTI (.nii) files
   # You'll need to convert to PNG slices
   # Tools: nibabel, SimpleITK
   ```

3. **Organize files**:
   ```
   data/
     train/
       images/
         case_0000.png
         case_0001.png
       masks/
         case_0000.png
         case_0001.png
     test/
       images/
       masks/
   ```

### Option 2: Other Public Datasets

**COVID-19 CT Segmentation**:
- Search Kaggle for "COVID-19 CT segmentation"
- Download and organize similarly

**Grand Challenge**:
- Visit: https://grand-challenge.org/
- Many public medical imaging datasets
- Download and convert to PNG format

**TCIA (The Cancer Imaging Archive)**:
- Visit: https://www.cancerimagingarchive.net/
- Large collection of medical images

### Option 3: Your Own Data

If you have your own medical images:

1. **Prepare images**:
   - Convert to PNG format (or modify dataloader for other formats)
   - Grayscale (1 channel) or RGB (3 channels)
   - Consistent size (or modify network for variable sizes)

2. **Create masks**:
   - Use annotation tools (ITK-SNAP, 3D Slicer, etc.)
   - Binary masks: 0 = background, 1 = lesion
   - Multi-class: 0, 1, 2, ... for different structures
   - Same size as corresponding images

3. **Update configuration**:
   ```python
   # In workflow JSON or node config:
   in_channels: 1  # for grayscale, 3 for RGB
   out_channels: 2  # for binary, N for N-class
   ```

---

## Re-creating Synthetic Data

If you want to re-generate synthetic data with different parameters:

```bash
python download_dataset.py
```

Then select:
- Option 1: Create synthetic dataset
- Enter number of training samples (default: 50)
- Enter number of test samples (default: 20)

---

## Summary

| Aspect | Current (Synthetic) | Real Medical Data |
|--------|-------------------|-------------------|
| **Data Type** | Random noise + circles | CT/MRI scans |
| **Anatomy** | None | Brain, organs, etc. |
| **Difficulty** | Very Easy | Medium to Hard |
| **Purpose** | Demo/Testing | Clinical/Research |
| **Metrics** | 0.95+ Dice | 0.70-0.90 Dice |
| **Clinical Value** | None | High |
| **Training Samples** | 50 | 100-1000+ |
| **Image Quality** | Synthetic | Real medical grade |

---

## Next Steps

### To Continue with Synthetic Data (Learning)

✓ **Good for**:
- Understanding the framework
- Testing different models
- Experimenting with parameters
- Learning the GUI

**Keep using**: Current setup is fine

### To Work with Real Data (Production)

✓ **Required for**:
- Research publications
- Clinical applications
- Meaningful comparisons
- Real insights

**Steps**:
1. Download real medical imaging dataset
2. Replace files in `data/` directory
3. Update network config if needed
4. Re-train and evaluate
5. Compare to published baselines

---

## FAQ

**Q: Why don't the images look like real scans?**
A: Because they're not! They're synthetic data created with random noise and circles.

**Q: Can I publish results from this data?**
A: No - it's synthetic demo data with no clinical relevance.

**Q: Why is the accuracy so high?**
A: The task (finding circles in noise) is trivial compared to real medical imaging.

**Q: How do I use my own data?**
A: Replace files in `data/train/` and `data/test/` with your images and masks.

**Q: What format should my data be?**
A: PNG files work by default. Modify the dataloader for other formats (DICOM, NIfTI, etc.).

**Q: The visualizations look strange - is something wrong?**
A: No! They're correctly showing the synthetic data. Real data would look very different.

---

For detailed explanations, see:
- `docs/data/SYNTHETIC_DATASET_EXPLAINED.md`
