# What to Expect When Loading Workflows in GUI

## 🎨 GUI Layout

When you launch the GUI, you'll see:

```
┌─────────────────────────────────────────────────────────┐
│ Medical Imaging Framework - Workflow Editor            │
├──────────────┬──────────────────────────────┬───────────┤
│              │                              │           │
│  Node        │     (Empty Canvas)           │ Controls  │
│  Library     │                              │           │
│              │   This area is currently     │ Workflow  │
│ DATA         │   not used for visual        │ Info:     │
│ • DataLoader │   display in this basic      │           │
│ • ImagePath..│   prototype.                 │ Nodes: 0  │
│              │                              │ Links: 0  │
│ NETWORKS     │   Loaded nodes appear in     │           │
│ • UNet2D     │   the Controls panel →       │           │
│ • UNet3D     │                              │ [Execute] │
│ ...          │                              │ [Validate]│
│              │                              │ [Save]    │
│ (24 nodes    │                              │ [Load]    │
│  total)      │                              │           │
└──────────────┴──────────────────────────────┴───────────┘
```

## ✅ After Loading `training_workflow.json`

### 1. Popup Message

You'll see a popup with:

```
✓ Workflow loaded successfully!

File: training_workflow.json
Nodes: 3
Connections: 0

Loaded Nodes:
  • data_loader (MedicalSegmentationLoader)
  • unet_model (UNet2D)
  • loss_function (LossFunction)

Note: Check the 'Controls' panel (right side) for detailed node list.

To execute this workflow, use the Python script:
python examples/medical_segmentation_pipeline/train_pipeline.py
```

Click **OK** to close this popup.

### 2. Controls Panel (Right Side)

The **Controls** panel will update to show:

```
Workflow Information
────────────────────
Workflow: Medical Segmentation Training
Nodes: 3
Connections: 0

Loaded Nodes:
  • data_loader (MedicalSegmentationLoader)
  • unet_model (UNet2D)
  • loss_function (LossFunction)
```

### 3. Status Bar (Bottom)

The status bar shows:
```
Workflow loaded: /path/to/training_workflow.json
```

## 🔍 What You WON'T See

❌ **No visual node graph on the canvas**
- The canvas area (center) remains empty
- This is a basic GUI prototype
- Visual node rendering is not implemented yet

❌ **No connection lines**
- Even though connections are defined in JSON
- They're stored in memory but not drawn

❌ **No drag-and-drop**
- Can't move nodes around
- Can't create connections visually

## ✅ What You CAN Do

1. **Browse Node Library** (Left Panel)
   - See all 24 available nodes
   - Read node descriptions (hover over buttons)

2. **View Loaded Workflow** (Right Panel - Controls)
   - See which nodes were loaded
   - See node names and types
   - Check number of connections

3. **Validate Workflow**
   - Click "✓ Validate Workflow" button
   - Check for configuration issues

4. **View Workflow Info**
   - The Controls panel shows all loaded nodes
   - Scroll to see complete list if many nodes

## 🏃 To Actually Run the Pipeline

The GUI is for **viewing** the workflow structure only.

To **execute** the training pipeline:

```bash
# In a terminal:
cd /home/jinhyeongpark/Codes/Node_DL_MedicalImaging
python examples/medical_segmentation_pipeline/train_pipeline.py
```

This will:
- Load the same nodes
- Connect them properly
- Execute the training
- Save the trained model

## 📊 Testing Workflow (`testing_workflow.json`)

When you load `testing_workflow.json`, you'll see:

### Popup Message:
```
✓ Workflow loaded successfully!

File: testing_workflow.json
Nodes: 4
Connections: 0

Loaded Nodes:
  • data_loader (MedicalSegmentationLoader)
  • unet_model (UNet2D)
  • predictor (BatchPredictor)
  • metrics (MetricsCalculator)
```

### Controls Panel:
```
Workflow: Medical Segmentation Testing
Nodes: 4
Connections: 0

Loaded Nodes:
  • data_loader (MedicalSegmentationLoader)
  • unet_model (UNet2D)
  • predictor (BatchPredictor)
  • metrics (MetricsCalculator)
```

## 💡 Summary

| Feature | Status |
|---------|--------|
| Load workflow JSON | ✅ Works |
| View node list | ✅ Works (in Controls panel) |
| Browse node library | ✅ Works |
| Validate workflow | ✅ Works |
| Visual node display | ❌ Not implemented |
| Visual connections | ❌ Not implemented |
| Drag and drop | ❌ Not implemented |
| Execute from GUI | ⚠️ Use Python scripts instead |

## 🎯 Recommended Workflow

1. **Explore in GUI**
   ```bash
   python examples/medical_segmentation_pipeline/launch_gui.py
   ```
   - Load workflows to see structure
   - Browse available nodes
   - Understand the pipeline

2. **Execute with Python**
   ```bash
   python examples/medical_segmentation_pipeline/train_pipeline.py
   python examples/medical_segmentation_pipeline/test_pipeline.py
   ```
   - Actual training and testing
   - Full functionality
   - Progress monitoring
   - Result visualization

## 🐛 Troubleshooting

**Q: The center canvas is empty - is this normal?**
A: Yes! The GUI doesn't draw nodes visually yet. Check the Controls panel on the right to see loaded nodes.

**Q: Can I click Execute Workflow in the GUI?**
A: You can try, but it's better to use the Python scripts for reliable execution with proper connections.

**Q: How do I see node details?**
A: Check the Controls panel (right side) - it lists all loaded nodes with their types.

**Q: The workflow has 0 connections?**
A: Connections are set up in the Python scripts. The JSON files define nodes but execute() methods handle connections.

---

**The GUI is a prototype for browsing and understanding the framework structure. For actual pipeline execution, use the Python scripts!** 🚀
