# Segmentation Overlay Visualization

## Overview

The **SegmentationOverlay** node creates visual overlays of ground truth and model predictions on input medical images. This helps you:
- Visually inspect segmentation quality
- Compare predictions with ground truth
- Identify error patterns (false positives/negatives)
- Create publication-ready figures

All visualizations are automatically saved to disk with a configurable output directory.

---

## Features

### What It Creates

1. **Individual Overlay Images** - One file per input image with 4 panels:
   - Original image
   - Ground truth overlay (green)
   - Prediction overlay (red)
   - Comparison overlay (green + red + yellow for overlap)

2. **Grid Visualization** - Single file showing multiple images in a grid

3. **Legend** - Color key explaining the overlay colors

### Color Coding

- **Green**: Ground truth segmentation
- **Red**: Model prediction
- **Yellow**: Correct prediction (overlap between GT and prediction)
- **Green only**: False negative (missed by model)
- **Red only**: False positive (incorrectly predicted)

---

## Configuration Parameters

### output_dir
- **Type**: Text field
- **Default**: `visualization_output`
- **Description**: Directory where visualizations will be saved
- **Examples**:
  - `visualization_output` - relative path
  - `examples/medical_segmentation_pipeline/visualization_output` - nested path
  - `/absolute/path/to/output` - absolute path
- **Note**: Directory is created automatically if it doesn't exist

### max_images
- **Type**: Text field (integer)
- **Default**: `10`
- **Description**: Maximum number of images to visualize
- **Recommended**:
  - `5-10` for quick inspection
  - `20-50` for thorough review
  - `100+` for full dataset (but takes longer)

### alpha
- **Type**: Text field (float)
- **Default**: `0.4`
- **Range**: 0.0 to 1.0
- **Description**: Transparency of overlay masks
- **Effect**:
  - `0.0` = Completely transparent (only see image)
  - `0.4` = Balanced visibility (recommended)
  - `0.7` = More opaque overlay
  - `1.0` = Completely opaque (can't see image underneath)

### save_individual
- **Type**: Choice (True/False)
- **Default**: `True`
- **Description**: Whether to save individual overlay images for each input
- **Use True**: When you want to inspect each image in detail
- **Use False**: When you only want the grid view

### save_grid
- **Type**: Choice (True/False)
- **Default**: `True`
- **Description**: Whether to save a grid visualization
- **Use True**: For quick overview of multiple images
- **Use False**: When you only need individual images

---

## Input Requirements

The SegmentationOverlay node requires three inputs:

### images (TENSOR)
- **Type**: Float tensor
- **Shape**: `(batch, channels, height, width)` or `(batch, height, width)`
- **Description**: Original input images
- **Source**: `predictor.all_images` output port

### labels (TENSOR)
- **Type**: Long tensor
- **Shape**: `(batch, height, width)`
- **Description**: Ground truth segmentation masks
- **Values**: Integer class labels (0, 1, 2, ...)
- **Source**: `predictor.all_labels` output port

### predictions (TENSOR)
- **Type**: Long tensor
- **Shape**: `(batch, height, width)`
- **Description**: Model predictions
- **Values**: Integer class labels (0, 1, 2, ...)
- **Source**: `predictor.all_predictions` output port

---

## Workflow Integration

### Updated BatchPredictor

The BatchPredictor node now outputs:
- `all_predictions` - Model predictions
- `all_labels` - Ground truth labels
- `all_images` - **NEW**: Input images for visualization

### Connection Pattern

```
predictor.all_images      → visualization.images
predictor.all_labels      → visualization.labels
predictor.all_predictions → visualization.predictions
```

---

## Example: Testing Workflow

Here's the complete workflow from `testing_workflow.json`:

```
┌──────────────┐
│ data_loader  │
└──────┬───────┘
       │
       │    ┌──────────────┐
       │    │  unet_model  │
       │    └──────┬───────┘
       │           │
       │    ┌──────────────────┐
       │    │ checkpoint_loader│
       │    └──────┬───────────┘
       │           │
       ▼           ▼
   ┌─────────────────┐
   │   predictor     │
   └──┬───┬───┬──────┘
      │   │   │
      │   │   └──────────────┐
      │   │                  │
      │   │  all_images      │  all_predictions
      │   │  all_labels      │
      ▼   ▼                  ▼
   ┌──────────┐      ┌──────────────────┐
   │ metrics  │      │  visualization   │
   └──────────┘      └──────────────────┘
                     Saves overlays to disk
```

---

## Output Files

When you execute a testing workflow with visualization enabled:

### Directory Structure

```
examples/medical_segmentation_pipeline/
└── visualization_output/
    ├── overlay_0000.png    ← Image 0 with 4 panels
    ├── overlay_0001.png    ← Image 1 with 4 panels
    ├── overlay_0002.png    ← Image 2 with 4 panels
    ├── ...
    ├── overlay_grid.png    ← Grid showing all images
    └── legend.png          ← Color legend
```

### Individual Overlay Format

Each `overlay_XXXX.png` contains 4 panels side-by-side:

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   Original  │ Ground Truth│ Prediction  │ Comparison  │
│    Image    │  (Green)    │   (Red)     │(GT+Pred)    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### Grid Visualization

`overlay_grid.png` shows up to 16 images in a 4×4 grid, each with the combined comparison overlay.

---

## Configuration in GUI

### Method 1: Edit Existing Node

1. Load `testing_workflow.json` in GUI
2. Double-click the **visualization** node
3. Modify parameters:
   - **Output Directory**: Change to your desired path
   - **Max Images**: Set how many to visualize
   - **Alpha**: Adjust transparency
4. Click OK
5. Save workflow (Ctrl+S)

### Method 2: Add to Custom Workflow

1. From Node Library, add **SegmentationOverlay** node
2. Connect three inputs from BatchPredictor:
   - `all_images` → `images`
   - `all_labels` → `labels`
   - `all_predictions` → `predictions`
3. Configure parameters
4. Execute workflow

---

## Interpreting Visualizations

### Good Segmentation

**Characteristics**:
- Lots of **yellow** (correct predictions)
- Minimal **green only** (few false negatives)
- Minimal **red only** (few false positives)
- Prediction shape closely matches ground truth

**Example interpretation**:
```
✓ Yellow dominates → Most of lesion correctly detected
✓ Little green showing → Few missed pixels
✓ Little red showing → Few false alarms
→ Excellent segmentation!
```

### Over-Segmentation (Too Liberal)

**Characteristics**:
- Moderate **yellow** (some correct)
- Lots of **red only** (many false positives)
- Little **green only** (few false negatives)
- Prediction extends beyond ground truth

**Problem**: Model predicts lesions too aggressively

**Solution**:
- Increase classification threshold
- Adjust loss function weights
- Add regularization

### Under-Segmentation (Too Conservative)

**Characteristics**:
- Moderate **yellow** (some correct)
- Lots of **green only** (many false negatives)
- Little **red only** (few false positives)
- Prediction smaller than ground truth

**Problem**: Model is too conservative, misses lesion boundaries

**Solution**:
- Decrease classification threshold
- Increase weight of positive class in loss
- Train longer

### Poor Segmentation

**Characteristics**:
- Little **yellow** (few correct predictions)
- Mix of **green only** and **red only**
- Prediction shape doesn't match ground truth
- Random-looking patterns

**Problem**: Model hasn't learned properly

**Solution**:
- Check data loading
- Verify model architecture
- Adjust learning rate
- Train from checkpoint
- Increase training epochs

---

## Tips and Best Practices

### Choosing max_images

✓ **For debugging**: `max_images: 5-10`
- Quick to generate
- Shows representative samples
- Good for iterative development

✓ **For thorough review**: `max_images: 20-50`
- More comprehensive view
- Catches edge cases
- Still manageable to review

✓ **For full dataset**: `max_images: 100+`
- Complete visualization
- Takes longer to generate
- Creates many files

### Adjusting Transparency (alpha)

**alpha = 0.3**: Very subtle overlay
- Good when image details are important
- Harder to see segmentation

**alpha = 0.4** (default): Balanced
- Good compromise
- Both image and overlay visible

**alpha = 0.6**: Strong overlay
- Segmentation clearly visible
- Image details still show through

**alpha = 0.8**: Dominant overlay
- Focus on segmentation
- Image less visible

### Organizing Output

**Strategy 1: Experiment-specific directories**
```
output_dir: "results/experiment_1/visualizations"
output_dir: "results/experiment_2/visualizations"
```

**Strategy 2: Date-based directories**
```
output_dir: "visualizations/2026-01-31"
```

**Strategy 3: Checkpoint-specific**
```
output_dir: "visualizations/epoch_10"
output_dir: "visualizations/best_model"
```

---

## Common Issues

### Issue: No visualizations created

**Possible causes**:
- Visualization node not connected
- Missing required inputs
- Output directory permission denied

**Solutions**:
1. Check all 3 connections are present
2. Verify predictor executed successfully
3. Use absolute path for output_dir
4. Check file permissions

### Issue: Images look wrong

**Possible causes**:
- Data not normalized properly
- Wrong tensor shape
- Incorrect color mapping

**Solutions**:
1. Check image tensor values are reasonable
2. Verify tensor shapes match expected format
3. Check console output for warnings

### Issue: Grid is too small to see

**Solution**: Individual overlays show more detail. Use those for close inspection.

### Issue: Too many files generated

**Solution**:
- Reduce `max_images`
- Set `save_individual: False` if you only need grid
- Clean up old visualizations periodically

---

## Example Configurations

### Minimal Configuration (Quick Check)

```json
{
  "output_dir": "quick_check",
  "max_images": "5",
  "alpha": "0.4",
  "save_individual": "False",
  "save_grid": "True"
}
```

**Result**: One grid image with 5 samples

### Detailed Analysis

```json
{
  "output_dir": "detailed_analysis",
  "max_images": "20",
  "alpha": "0.5",
  "save_individual": "True",
  "save_grid": "True"
}
```

**Result**: 20 individual images + 1 grid + legend

### Full Dataset Review

```json
{
  "output_dir": "full_dataset_viz",
  "max_images": "100",
  "alpha": "0.4",
  "save_individual": "True",
  "save_grid": "False"
}
```

**Result**: 100 individual images (no grid for performance)

### Publication Figures

```json
{
  "output_dir": "publication_figures",
  "max_images": "3",
  "alpha": "0.5",
  "save_individual": "True",
  "save_grid": "False"
}
```

**Result**: High-quality individual overlays for paper figures

---

## Workflow Execution Output

When you run the testing workflow, you'll see:

```
============================================================
SEGMENTATION VISUALIZATION
============================================================
Output directory: /path/to/visualization_output
Creating visualizations for 10 images...
✓ Saved 10 individual overlay images
✓ Saved grid visualization with 10 images
✓ Saved legend
============================================================
```

---

## Related Documentation

- [Testing Guide](../testing/LOADING_TRAINED_MODELS.md) - Complete testing workflow
- [Metrics Explained](../testing/SEGMENTATION_METRICS_EXPLAINED.md) - Understanding segmentation metrics
- [Port Types](../gui/PORT_TYPES_GUIDE.md) - Understanding data flow

---

## Summary

**To add visualization to your testing workflow**:

1. Ensure BatchPredictor outputs `all_images` (already updated)
2. Add SegmentationOverlay node to workflow
3. Connect 3 ports from predictor to visualization
4. Configure `output_dir` parameter
5. Execute workflow
6. Inspect saved images in output directory

**Key benefits**:
- Visual quality assessment
- Error pattern identification
- Quick comparison of GT vs prediction
- Publication-ready figures

**Color legend**:
- **Green** = Ground truth only (false negative)
- **Red** = Prediction only (false positive)
- **Yellow** = Overlap (correct)
