# Quick Start: Medical Segmentation Pipeline with GUI

## 🚀 Launch the GUI

```bash
cd /home/jinhyeongpark/Codes/Node_DL_MedicalImaging
python examples/medical_segmentation_pipeline/launch_gui.py
```

## 📂 Load a Workflow

In the GUI:

1. **File → Load Workflow** (Ctrl+O)
2. Select: `training_workflow.json` or `testing_workflow.json`
3. Click **Open**

You'll see:
- ✅ Popup showing loaded nodes
- ✅ Controls panel (right) listing all nodes
- ❌ Canvas (center) remains empty - this is a basic GUI prototype

**See [GUI_WHAT_TO_EXPECT.md](GUI_WHAT_TO_EXPECT.md) for detailed explanation of what you'll see.**

## 👁️ What You Can Do in the GUI

✅ **View Pipeline Structure**
- See all nodes in the workflow
- Understand the architecture visually

✅ **Browse Node Library**
- Explore 24 available nodes
- Check node descriptions

✅ **View Configurations**
- Click any node to see its settings
- View input/output ports

✅ **Validate Workflow**
- Workflow → Validate (Ctrl+V)
- Check for configuration issues

## 🏃 Running the Pipeline

**The GUI is for visualization.** To actually execute the pipeline, use the Python scripts:

### Training
```bash
python examples/medical_segmentation_pipeline/train_pipeline.py
```

This will:
- Load the 50 training samples
- Train UNet2D for 20 epochs
- Save best model to `models/`
- Takes ~5-10 minutes

### Testing
```bash
python examples/medical_segmentation_pipeline/test_pipeline.py
```

This will:
- Load trained model
- Run inference on 20 test samples
- Calculate metrics (Dice, IoU, etc.)
- Generate comparison visualizations
- Save results to `results/`

## 📊 Workflows Available

### training_workflow.json
- **MedicalSegmentationLoader**: Custom data loader
- **UNet2D**: 2D U-Net model (3 levels, 32 channels)
- **LossFunction**: Dice loss for segmentation

### testing_workflow.json
- **MedicalSegmentationLoader**: Data loader
- **UNet2D**: Trained model
- **BatchPredictor**: Inference engine
- **MetricsCalculator**: Evaluation metrics

## 🎯 Quick Workflow

```bash
# 1. Launch GUI and explore
python examples/medical_segmentation_pipeline/launch_gui.py

# 2. Train the model (in another terminal)
python examples/medical_segmentation_pipeline/train_pipeline.py

# 3. Test and visualize (after training)
python examples/medical_segmentation_pipeline/test_pipeline.py

# 4. View results
ls examples/medical_segmentation_pipeline/results/visualizations/
```

## 📖 More Information

- **[GUI_GUIDE.md](GUI_GUIDE.md)** - Complete GUI documentation
- **[README.md](README.md)** - Full pipeline documentation
- **[../../docs/INDEX.md](../../docs/INDEX.md)** - Framework documentation

## 💡 Key Points

- GUI shows **24 nodes** (23 framework + 1 custom)
- Workflows are **JSON files** that define node configurations
- GUI is for **visualization and exploration**
- Python scripts are for **execution**
- Dataset already generated (50 train + 20 test samples)

---

**Happy exploring!** 🎨
