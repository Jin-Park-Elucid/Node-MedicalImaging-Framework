# Visual GUI - Complete Guide 🎨

## ✨ NEW: Nodes Are Now Drawn Visually!

The GUI now features **full visual node rendering** with an interactive canvas!

## 🚀 Launch the Visual GUI

```bash
cd /home/jinhyeongpark/Codes/Node_DL_MedicalImaging
python examples/medical_segmentation_pipeline/launch_gui.py
```

## 🎨 What You'll See

### Main Window Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ File   View   Workflow   Help                    [Menu Bar]     │
├──────────────┬──────────────────────────────────────┬───────────┤
│              │                                      │           │
│  Node        │        Visual Canvas                 │ Controls  │
│  Library     │     ┌─────────────────┐             │           │
│              │     │ data_loader     │             │ Workflow  │
│ DATA         │     │ (Loader)        │             │ Info      │
│ • DataLoader │     │ In: 0 | Out: 4  │             │           │
│ • ImagePath..│     └────┬────────────┘             │ Nodes: 3  │
│   ...        │          │ curved line              │ Links: 0  │
│              │          ▼                           │           │
│ NETWORKS     │     ┌─────────────────┐             │ • data... │
│ • UNet2D ←   │     │ unet_model      │             │ • unet... │
│ • UNet3D     │     │ (UNet2D)        │             │ • loss... │
│   ...        │     │ In: 1 | Out: 1  │             │           │
│              │     └─────────────────┘             │ [Execute] │
│ TRAINING     │                                      │ [Validate]│
│ • Trainer    │     ┌─────────────────┐             │ [Save]    │
│ • Loss...    │     │ loss_function   │             │ [Load]    │
│   ...        │     │ (LossFunction)  │             │           │
│              │     │ In: 0 | Out: 1  │             │           │
└──────────────┴─────└─────────────────┘─────────────┴───────────┘
│ Status: Workflow loaded - 3 nodes displayed                     │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Visual Features

### Node Appearance

Each node is displayed as a **rounded rectangle** with:

- **Title Bar** (top) - Shows node name
  - Color-coded by category:
    - 🔵 Blue: Data nodes
    - 🟢 Green: Network nodes
    - 🔴 Red: Training nodes
    - 🟠 Orange: Inference nodes
    - 🟣 Purple: Visualization nodes

- **Body** (center) - Shows node type and port counts
  - Node type (e.g., "UNet2D", "Loader")
  - Port summary: "In: X | Out: Y"

- **Ports** (sides)
  - 🔵 Blue circles on left: Input ports
  - 🟠 Orange circles on right: Output ports
  - Hover over ports to see names and types

- **Visual Effects**
  - Drop shadow for depth
  - Gradient fill
  - Highlights when selected (yellow border)
  - Highlights when hovered (bright border)

### Connections

Connections between nodes appear as:
- **Curved lines** from output port to input port
- **Arrow heads** pointing to the target
- Automatically update when nodes are moved

### Example Visual

```
┌─────────────────────┐
│  data_loader        │ ← Title (node name)
│─────────────────────│
│  MedicalSegLoader   │ ← Type
│  In: 0 | Out: 4     │ ← Port counts
│                     │
│  ● train_loader ────┼──→ ● (curved connection line)
│  ● test_loader      │
│  ● num_train        │
│  ● num_test         │
└─────────────────────┘
```

## 🖱️ Mouse Controls

### Navigation

| Action | Control |
|--------|---------|
| **Pan** | Left click + drag on canvas |
| **Zoom In** | Mouse wheel up |
| **Zoom Out** | Mouse wheel down |
| **Fit to View** | Ctrl+0 or View → Fit to View |
| **Reset View** | View → Reset View |

### Node Interaction

| Action | Control |
|--------|---------|
| **Select Node** | Left click on node |
| **Move Node** | Click and drag node |
| **Node Info** | Right click → Node Info |
| **View Config** | Right click → View Configuration |
| **Delete Node** | Right click → Delete Node |

### Port Interaction

| Action | Result |
|--------|--------|
| **Hover over port** | See port name and data type |
| **Port colors** | Blue = Input, Orange = Output |

## ⌨️ Keyboard Shortcuts

### File Operations
- `Ctrl+N` - New workflow (clear all)
- `Ctrl+O` - Load workflow
- `Ctrl+S` - Save workflow (with node positions)
- `Ctrl+Q` - Quit

### View Controls
- `Ctrl++` - Zoom in
- `Ctrl+-` - Zoom out
- `Ctrl+0` - Fit to view

### Workflow
- `Ctrl+V` - Validate workflow
- `Ctrl+E` - Execute workflow
- `Ctrl+L` - Auto-layout nodes

## 📂 Loading a Workflow

1. **File → Load Workflow** (or `Ctrl+O`)
2. Select `training_workflow.json`
3. **Visual nodes appear on canvas!**

You'll see:
- ✅ All 3 nodes drawn as visual boxes
- ✅ Proper positioning
- ✅ Color-coded by category
- ✅ Connections shown (if any)

### After Loading

The popup shows:
```
✓ Workflow loaded and displayed!

File: training_workflow.json
Nodes: 3
Connections: 0

Loaded Nodes:
  • data_loader (MedicalSegmentationLoader)
  • unet_model (UNet2D)
  • loss_function (LossFunction)

Visual Features:
• Drag nodes to reposition
• Click nodes to select
• Right-click for context menu
• Mouse wheel to zoom
```

## 🎨 Creating Nodes

### From Node Library

1. Click a node button in the left panel (e.g., "UNet2D")
2. Node appears in center of view
3. Drag to reposition

### Colors by Category

- **DATA** (Blue): DataLoader, ImagePathLoader, etc.
- **NETWORKS** (Green): UNet2D, UNet3D, ResNet, etc.
- **TRAINING** (Red): Trainer, LossFunction, Optimizer
- **INFERENCE** (Orange): Predictor, MetricsCalculator
- **VISUALIZATION** (Purple): ImageViewer, MetricsPlotter

## 💾 Saving with Positions

When you save a workflow:
- Node positions are preserved
- Next time you load, nodes appear in same positions
- Reorganize your workflow visually, then save!

## 🎯 Context Menu (Right-Click)

Right-click any node to see:

### Node Info
Shows:
- Node name and type
- Category
- All input ports with types
- All output ports with types

### View Configuration
Shows all config parameters:
```
Configuration for unet_model:

  in_channels: 1
  out_channels: 2
  base_channels: 32
  depth: 3
```

### Delete Node
Removes node from canvas and workflow

## 🎨 Visual Examples

### Training Workflow Display

When you load `training_workflow.json`:

```
     ┌─────────────────────┐
     │   data_loader       │  (Steel Blue)
     │   MedicalSegLoader  │
     └──────────┬──────────┘
                │
                │
     ┌──────────▼──────────┐
     │   unet_model        │  (Forest Green)
     │   UNet2D            │
     └─────────────────────┘

     ┌─────────────────────┐
     │   loss_function     │  (Steel Blue)
     │   LossFunction      │
     └─────────────────────┘
```

### Testing Workflow Display

When you load `testing_workflow.json`:

```
     ┌─────────────┐
     │ data_loader │ ────┐
     └─────────────┘     │
                         ▼
     ┌─────────────┐   ┌───────────┐   ┌──────────┐
     │ unet_model  │──→│ predictor │──→│ metrics  │
     └─────────────┘   └───────────┘   └──────────┘
```

## 🔧 Advanced Features

### Auto-Layout

If nodes are cluttered:
1. **Workflow → Auto-Layout Nodes** (or `Ctrl+L`)
2. Nodes automatically arrange in a grid
3. Adjust positions manually after

### Zoom and Navigation

- Start zoomed to fit all nodes
- Zoom in to see details
- Pan around large workflows
- Use `Ctrl+0` to fit everything back

### Selection

- Click node to select (yellow border)
- Selected node shows in front
- Deselect by clicking canvas

## 🎯 Complete Workflow

### Visual Design Workflow

1. **Launch GUI**
   ```bash
   python examples/medical_segmentation_pipeline/launch_gui.py
   ```

2. **Load or Create**
   - Load existing: `Ctrl+O` → select JSON
   - Create new: Click nodes from library

3. **Arrange Visually**
   - Drag nodes to desired positions
   - Use zoom/pan for navigation
   - Right-click to view node details

4. **Save Layout**
   - `Ctrl+S` to save with positions
   - Positions preserved for next load

5. **Execute**
   - Use Python scripts for actual execution
   - GUI is for visualization and design

## 📊 Color-Coding Reference

| Category | Color | Example Nodes |
|----------|-------|---------------|
| Data | 🔵 Steel Blue | DataLoader, ImagePathLoader |
| Networks | 🟢 Forest Green | UNet2D, UNet3D, ResNet |
| Training | 🔴 Crimson | Trainer, LossFunction, Optimizer |
| Inference | 🟠 Dark Orange | Predictor, MetricsCalculator |
| Visualization | 🟣 Blue Violet | ImageViewer, MetricsPlotter |

## 🐛 Troubleshooting

**Q: Canvas is still empty?**
A: Make sure to load the workflow after launching. The canvas starts empty.

**Q: Can't see nodes?**
A: Press `Ctrl+0` to fit all nodes in view.

**Q: Nodes too small/big?**
A: Use mouse wheel to zoom in/out.

**Q: Can't move nodes?**
A: Make sure you're clicking directly on the node (not the canvas background).

**Q: Want to reset layout?**
A: Use `Ctrl+L` for auto-layout or manually drag nodes.

## 🚀 Try It Now!

```bash
# Launch the visual GUI
python examples/medical_segmentation_pipeline/launch_gui.py

# Load a workflow
# File → Load Workflow → training_workflow.json

# Enjoy the visual node graph! 🎨
```

## 📚 Related Documentation

- **[README.md](README.md)** - Pipeline overview
- **[GUI_GUIDE.md](GUI_GUIDE.md)** - General GUI guide
- **[QUICKSTART_GUI.md](QUICKSTART_GUI.md)** - Quick start

---

**Visual node rendering implemented!** 🎉

The GUI now provides a full interactive visual experience for designing and understanding your medical imaging pipelines!
