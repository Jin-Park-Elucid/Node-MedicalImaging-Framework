# Editing Node Parameters

## Overview

You can modify important parameters of any node in the GUI through an interactive parameter editing dialog. This allows you to configure nodes like adjusting learning rates, batch sizes, number of epochs, and other settings without manually editing JSON files.

---

## How to Edit Parameters

### Method 1: Double-Click (Quick Access)

1. **Double-click** any node in the graph
2. Parameter editing dialog opens automatically
3. Modify values as needed
4. Click **OK** to save changes

### Method 2: Context Menu (Right-Click)

1. **Right-click** on a node
2. Select **"⚙ Edit Parameters"** from the menu
3. Modify values in the dialog
4. Click **OK** to save changes

---

## Parameter Dialog Features

### Field Types

The dialog automatically creates appropriate widgets based on parameter types:

#### Text Fields
- Used for: Numbers, strings, file paths
- Examples: `learning_rate`, `batch_size`, `num_epochs`
- Simply type the new value

#### Choice Fields (Dropdowns)
- Used for: Predefined options
- Examples: `optimizer_type` (adam/sgd/adamw), `loss_type` (cross_entropy/dice/mse)
- Select from available options

### Current Values
- All fields display current configured values
- If no value is set, shows the default value
- Values persist when you save the workflow

### Change Detection
- The dialog compares new values with current values
- **Only shows confirmation if parameters actually changed**
- If you click OK without changing anything, no confirmation appears

---

## Examples

### Training Node Parameters

**TrainerNode** has these editable parameters:
- `num_epochs`: Number of training epochs (default: 10)
- `learning_rate`: Learning rate for optimization (default: 0.001)
- `loss_type`: Loss function - cross_entropy, dice, or mse

**Example workflow:**
1. Add a Trainer node
2. Double-click the node
3. Change `num_epochs` from 10 to 20
4. Change `learning_rate` from 0.001 to 0.0005
5. Select `dice` for `loss_type`
6. Click OK
7. Parameters are now saved

### Data Loader Parameters

**MedicalSegmentationLoader** has these parameters:
- `data_dir`: Path to dataset directory
- `batch_size`: Batch size for training (default: 4)
- `num_workers`: Number of worker processes (default: 0)
- `shuffle_train`: Whether to shuffle training data (True/False)

### Network Parameters

**UNet2DNode** has these parameters:
- `in_channels`: Number of input channels (default: 1)
- `out_channels`: Number of output classes (default: 2)
- `base_channels`: Base number of channels (default: 64)
- `depth`: Network depth (default: 4)
- `bilinear`: Use bilinear upsampling (True/False)

### Optimizer Parameters

**OptimizerNode** has these parameters:
- `optimizer_type`: Optimizer type (adam/sgd/adamw)
- `learning_rate`: Learning rate (default: 0.001)
- `weight_decay`: Weight decay for regularization (default: 0.0)
- `momentum`: Momentum for SGD (default: 0.9)

---

## Workflow

### Typical Parameter Editing Workflow

1. **Load or create workflow**
   - Load existing JSON workflow
   - Or add nodes from palette

2. **Configure each node**
   - Double-click node to edit parameters
   - Adjust values based on your requirements
   - Click OK to save

3. **Save workflow** (Ctrl+S)
   - Parameters are saved in the JSON file
   - Includes both node positions and configurations

4. **Execute workflow** (Ctrl+E)
   - Nodes use your configured parameters
   - Training runs with your settings

---

## Parameter Validation

### Type Conversion
- Text fields are stored as strings
- Nodes handle type conversion internally
- Example: `"10"` (string) → `10` (int) in TrainerNode

### Invalid Values
- If you enter invalid values, nodes may fail during execution
- Example: Setting `batch_size` to `"abc"` will cause an error
- Check execution output for validation errors

### Required vs Optional
- All parameters shown in the dialog are optional
- If not set, nodes use default values
- Defaults are defined in each node's `get_field_definitions()`

---

## Tips

### Best Practices

✓ **Start with defaults**: Use default values initially, then tune based on results

✓ **Save frequently**: Use Ctrl+S to save workflow after parameter changes

✓ **Test incrementally**: Change one parameter at a time to understand its effect

✓ **Document settings**: Use meaningful workflow names to track different configurations

✓ **Smart confirmation**: The dialog only shows "Parameters Updated" if you actually changed something - click OK without changes to close silently

### Common Parameters to Adjust

**For Training:**
- `num_epochs`: More epochs = longer training, potentially better convergence
- `learning_rate`: Lower = slower but more stable, higher = faster but may diverge
- `batch_size`: Larger = faster training but more memory, smaller = slower but less memory

**For Data Loading:**
- `batch_size`: Match your GPU memory capacity
- `num_workers`: Set to 0 on Windows, 2-4 on Linux/Mac for faster loading
- `shuffle_train`: Usually True for training, False for reproducibility

**For Networks:**
- `base_channels`: More channels = more capacity but slower
- `depth`: Deeper networks = more capacity but harder to train

---

## Keyboard Shortcuts

- **Double-click node**: Open parameter dialog
- **Tab**: Move between fields in dialog
- **Enter**: Accept changes (when focused on a field)
- **Escape**: Cancel dialog

---

## Viewing Current Configuration

If you just want to see current parameters without editing:

1. Right-click node
2. Select **"View Configuration"**
3. Read-only dialog shows all current values

---

## Technical Details

### Where Parameters Are Stored

Parameters are stored in:
- **Memory**: `node.config` dictionary during GUI session
- **File**: JSON workflow file when saved

### Field Definitions

Each node defines its parameters in `get_field_definitions()`:

```python
def get_field_definitions(self):
    return {
        'num_epochs': {
            'type': 'text',
            'label': 'Number of Epochs',
            'default': '10'
        },
        'loss_type': {
            'type': 'choice',
            'label': 'Loss Function',
            'choices': ['cross_entropy', 'dice', 'mse'],
            'default': 'cross_entropy'
        }
    }
```

### Parameter Flow

1. User edits in dialog → `node.config` updated
2. Workflow saved → config written to JSON
3. Workflow loaded → config read from JSON → `node.config` populated
4. Node executes → reads values with `node.get_config('param_name', default)`

---

## Troubleshooting

### Dialog Doesn't Open

**Issue**: Double-click moves node instead of opening dialog

**Solution**: Click once to select, then double-click. Or use right-click menu instead.

### Changes Not Saved

**Issue**: Parameters reset after closing GUI

**Solution**: Save workflow (Ctrl+S) before closing GUI. Parameters are only persisted when you save.

### Node Fails After Parameter Change

**Issue**: Execution fails after changing parameters

**Solution**:
- Check parameter values are valid (numbers, paths exist, etc.)
- View error messages in execution output
- Reset to default values if unsure

### Parameter Not Showing in Dialog

**Issue**: Expected parameter doesn't appear in edit dialog

**Solution**:
- Node may not have `get_field_definitions()` implemented
- Parameter may be hard-coded in the node
- Use "View Configuration" to see all current config values

---

## Related Documentation

- [Creating Connections](CREATING_CONNECTIONS.md) - Connecting nodes
- [Port Types Guide](PORT_TYPES_GUIDE.md) - Understanding data types
- [Training vs Inference](TRAINING_VS_INFERENCE.md) - Workflow patterns

---

## Summary

**To edit node parameters:**
1. Double-click node or right-click → "Edit Parameters"
2. Modify values in dialog
3. Click OK to save
4. Save workflow (Ctrl+S) to persist changes

**Key benefits:**
- No manual JSON editing required
- Visual interface for all parameters
- Immediate feedback on configuration
- Changes persist in workflow files
