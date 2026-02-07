# Medical Segmentation Pipeline - GUI Guide

Quick guide for loading and running the medical segmentation pipeline in the GUI.

## Launch the GUI

From the project root:

```bash
python examples/medical_segmentation_pipeline/launch_gui.py
```

Or from the pipeline directory:

```bash
cd examples/medical_segmentation_pipeline
python launch_gui.py
```

This launches the GUI with the custom `MedicalSegmentationLoader` node registered.

## Load a Workflow

### Option 1: Training Workflow

1. Click **File → Load Workflow** (or press Ctrl+O)
2. Navigate to: `examples/medical_segmentation_pipeline/`
3. Select: `training_workflow.json`
4. Click **Open**

The GUI will load the training pipeline with:
- **MedicalSegmentationLoader** - Custom data loader
- **UNet2D** - Segmentation model
- **LossFunction** - Dice loss

**Note**: The nodes are visualized but not fully connected in the workflow file. For actual execution, use the Python script:
```bash
python examples/medical_segmentation_pipeline/train_pipeline.py
```

### Option 2: Testing Workflow

1. Click **File → Load Workflow**
2. Select: `testing_workflow.json`

The GUI will load the testing pipeline with:
- **MedicalSegmentationLoader** - Data loader
- **UNet2D** - Trained model
- **BatchPredictor** - Inference engine
- **MetricsCalculator** - Evaluation metrics

**Note**: For actual testing with visualizations, use:
```bash
python examples/medical_segmentation_pipeline/test_pipeline.py
```

## Workflow Controls

### Validate Workflow

Check the workflow structure:
- Click **Workflow → Validate** (or press Ctrl+V)
- View any configuration issues

### View Node Details

Explore node configurations:
- Click on any node to see its configuration
- View input/output ports
- Check parameter settings

### Execute Training/Testing

**For actual execution**, use the Python scripts instead of GUI execution:

**Training:**
```bash
python examples/medical_segmentation_pipeline/train_pipeline.py
```

**Testing:**
```bash
python examples/medical_segmentation_pipeline/test_pipeline.py
```

**Why use Python scripts?**
The GUI is excellent for:
- ✓ Visualizing pipeline structure
- ✓ Browsing available nodes
- ✓ Viewing configurations
- ✓ Understanding the architecture

The Python scripts provide:
- ✓ Full control over execution
- ✓ Proper node connections
- ✓ Progress monitoring
- ✓ Result visualization

### Save Workflow

Save modifications:
- Click **File → Save Workflow** (or press Ctrl+S)
- Choose location and filename

### Clear Workflow

Start fresh:
- Click **File → Clear Workflow** (or press Ctrl+N)

## GUI Layout

```
┌─────────────────────────────────────────────────────────┐
│ File    Workflow    Help              [Menu Bar]       │
├──────────────┬──────────────────────────────────────────┤
│              │                                          │
│  Node        │         Workflow Canvas                 │
│  Library     │      (Nodes and connections shown)      │
│              │                                          │
│ • Data       │                                          │
│ • Networks   │                                          │
│ • Training   │                                          │
│ • Inference  │                                          │
│ • Viz        │                                          │
│              │                                          │
├──────────────┼──────────────────────────────────────────┤
│ Workflow     │ Node Details                            │
│ Info         │ (Selected node configuration)           │
│              │                                          │
│ Nodes: 5     │                                          │
│ Links: 6     │                                          │
└──────────────┴──────────────────────────────────────────┘
│ Status: Ready                                          │
└────────────────────────────────────────────────────────┘
```

## Working with Nodes

### Browse Available Nodes

Left panel shows node categories:
- **Data**: Data loaders and augmentation
- **Networks**: Neural network architectures
- **Training**: Training components
- **Inference**: Prediction and evaluation
- **Visualization**: Display and plotting

### View Node Details

Click on a node in the canvas to see:
- Node type and name
- Input ports
- Output ports
- Configuration parameters

### Modify Node Configuration

To change node settings:
1. **Option A**: Edit the JSON workflow file directly
2. **Option B**: Clear workflow and rebuild with different configs

## Training Workflow Details

**Pipeline Structure:**
```
MedicalSegmentationLoader
    ├─→ train_loader → Trainer
    └─→ test_loader  → Trainer

UNet2D
    ├─→ module → Trainer
    └─→ module → Optimizer

LossFunction
    └─→ loss → Trainer

Optimizer
    └─→ optimizer → Trainer
```

**Execution Flow:**
1. Data loader creates train/test DataLoaders
2. UNet2D builds the model
3. Loss function creates Dice loss
4. Optimizer creates Adam with model parameters
5. Trainer runs training loop for 20 epochs
6. Best model saved to `models/best_model_*.pth`

## Testing Workflow Details

**Pipeline Structure:**
```
MedicalSegmentationLoader
    └─→ test_loader → BatchPredictor

UNet2D (with loaded weights)
    └─→ module → BatchPredictor

BatchPredictor
    ├─→ predictions → MetricsCalculator
    └─→ targets → MetricsCalculator

MetricsCalculator
    └─→ metrics → Print
```

**Note**: Load trained weights before executing testing workflow.

## Workflow Execution Tips

### Before Training

1. ✓ Ensure dataset exists: `examples/medical_segmentation_pipeline/data/`
2. ✓ Validate workflow (Ctrl+V)
3. ✓ Check console for any warnings
4. ✓ Ensure sufficient disk space for model checkpoints

### Before Testing

1. ✓ Train model first (or have `best_model_*.pth` in `models/`)
2. ✓ Update model path if needed
3. ✓ Validate workflow
4. ✓ Run `test_pipeline.py` separately for visualizations

### Monitoring Execution

During execution:
- Status bar shows current progress
- Console shows detailed logs
- Training progress updates each epoch
- Errors displayed in popup dialogs

## Advanced Usage

### Create Custom Workflow

1. Click **File → Clear Workflow**
2. Drag nodes from library to canvas
3. Click output port → drag → click input port to connect
4. Configure each node
5. Validate and execute

### Modify Existing Workflow

1. Load workflow JSON
2. Edit JSON file with text editor
3. Reload in GUI
4. Validate changes

### Add Custom Nodes

To add your own nodes to the GUI:
1. Create node class with `@NodeRegistry.register()` decorator
2. Import in `launch_gui.py` before launching GUI
3. Node appears automatically in library

## Troubleshooting

### "Node type not found" Error

**Problem**: MedicalSegmentationLoader not registered

**Solution**: Launch GUI using `launch_gui.py` (not direct GUI)

### Validation Fails

**Problem**: Missing connections or invalid configurations

**Solution**:
- Check all nodes have required connections
- Verify config values match expected types
- Review console error messages

### Execution Fails

**Problem**: Runtime error during workflow execution

**Solution**:
- Check console for detailed traceback
- Verify data paths are correct
- Ensure dataset exists
- Check CUDA availability if using GPU

### GUI Won't Open

**Problem**: Missing PyQt5 or dependencies

**Solution**:
```bash
source venv/bin/activate
pip install PyQt5
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+N | Clear workflow |
| Ctrl+O | Load workflow |
| Ctrl+S | Save workflow |
| Ctrl+V | Validate workflow |
| Ctrl+E | Execute workflow |
| Ctrl+Q | Quit application |

## File Locations

After execution, check these directories:

**Training Output:**
```
examples/medical_segmentation_pipeline/
└── models/
    ├── best_model_epoch_XX_loss_X.XXXX.pth
    └── checkpoint_epoch_XX.pth
```

**Testing Output:**
```
examples/medical_segmentation_pipeline/
└── results/
    ├── test_metrics.txt
    └── visualizations/
        └── comparison_*.png
```

## Next Steps

1. **Load and validate** training workflow
2. **Execute** training (takes 5-10 minutes)
3. **Check** saved model in `models/`
4. **Load** testing workflow
5. **Run** `test_pipeline.py` for visualizations
6. **View** results in `results/visualizations/`

## Related Documentation

- [Pipeline README](README.md) - Complete pipeline documentation
- [Framework GUI](../../docs/README.md#gui-features) - General GUI features
- [Creating Custom Nodes](../../docs/README.md#creating-custom-nodes) - Node development

---

**Quick Start:**
```bash
python launch_gui.py
# In GUI: File → Load Workflow → training_workflow.json
# Workflow → Execute
```
