# Testing the Connection Fix

## Changes Made

### 1. Enhanced Debugging in `gui/node_graphics.py`

Added comprehensive debug output to track:
- When `_create_connection()` is called
- Source and target ports being connected
- Whether editor is found
- Whether `graph.connect()` is called successfully
- Total number of links in graph after connection

### 2. Fixed Type Compatibility Check

Changed from strict equality to allow `DataType.ANY` to match with any type:

**Before:**
```python
if source_port.port.data_type != target_port.port.data_type:
```

**After:**
```python
if (source_port.port.data_type != target_port.port.data_type and
    source_port.port.data_type != DataType.ANY and
    target_port.port.data_type != DataType.ANY):
```

## How to Test

### Step 1: Completely Restart GUI

**CRITICAL**: You MUST restart the GUI completely to load the new debugging code:

```bash
# Close the GUI if it's running
# Then start fresh:
python examples/launch_gui.py
```

### Step 2: Create Nodes and Connect

1. Add these nodes:
   - UNet2D (will be named `UNet2D_0`)
   - Optimizer (will be named `Optimizer_1`)

2. Drag from `UNet2D_0.model` output port to `Optimizer_1.model` input port

3. **Watch the console** - you should see:
   ```
   [DEBUG] _create_connection called:
     Source: UNet2D_0.model (model)
     Target: Optimizer_1.model (model)
     → Looking for editor...
     → Found editor via self.scene().editor
     → Calling editor.graph.connect()...
     ✓ Connected: UNet2D_0.model -> Optimizer_1.model
     ✓ Graph now has 1 total links
   ```

### Step 3: Diagnose the Output

#### ✅ **Success Case** - You see all the debug messages ending with:
```
✓ Connected: UNet2D_0.model -> Optimizer_1.model
✓ Graph now has 1 total links
```
→ **Connection is working!** Continue to Step 4 to test execution.

#### ❌ **Problem Case 1** - You see:
```
[DEBUG] _create_connection called:
  ...
  → WARNING: Could not find editor!
  ⚠ WARNING: Could not find graph, creating connection without registering
```
→ **Problem**: Scene doesn't have editor reference. Check if `self.scene.editor = self` is in `editor.py` line 40.

#### ❌ **Problem Case 2** - You DON'T see ANY `[DEBUG]` messages
→ **Problem**: `_create_connection()` is not being called at all. The drag-and-drop might be failing earlier.

#### ❌ **Problem Case 3** - You see:
```
→ Cannot connect: incompatible types batch -> tensor
```
→ **Problem**: Port type mismatch. Check the PORT_TYPES_GUIDE.md for correct connection patterns.

### Step 4: Test Execution

After creating the connection:

1. Click "Execute Workflow"
2. You should see validation error for missing dataloader, loss_fn, etc. (expected - we only connected 2 nodes)
3. You should **NOT** see "Required input 'model' on node 'Optimizer_1' is not connected"

If you still see that error, the connection wasn't registered.

### Step 5: Test Save/Load

1. Save the workflow: File → Save Workflow → `test_debug.json`
2. Open the file and check:
   ```bash
   cat test_debug.json | python3 -m json.tool | grep -A 10 links
   ```
3. You should see:
   ```json
   "links": [
       {
           "source_node": "UNet2D_0",
           "source_port": "model",
           "target_node": "Optimizer_1",
           "target_port": "model"
       }
   ]
   ```

## What Each Debug Message Means

| Message | Meaning |
|---------|---------|
| `[DEBUG] _create_connection called:` | Mouse drag completed and method started |
| `Source: ... Target: ...` | Shows what you're trying to connect |
| `→ Looking for editor...` | Searching for editor reference |
| `→ Found editor via self.scene().editor` | ✅ Editor found correctly |
| `→ WARNING: Could not find editor!` | ❌ Scene doesn't have editor reference |
| `→ Calling editor.graph.connect()...` | About to register connection in graph |
| `✓ Connected: ...` | ✅ Connection registered successfully |
| `✓ Graph now has N total links` | Shows total connections in graph |
| `✗ Error creating connection: ...` | ❌ Exception occurred, with traceback |

## Common Issues and Solutions

### Issue: No debug output at all

**Cause**: GUI hasn't loaded the new code
**Solution**: Completely close and restart the GUI

### Issue: "Could not find editor"

**Cause**: `scene.editor` reference not set
**Solution**: Verify line 40 in `editor.py`:
```bash
grep -n "self.scene.editor = self" medical_imaging_framework/gui/editor.py
```
Should output: `40:        self.scene.editor = self`

### Issue: Connection created but not saved

**Cause**: Graph.connect() not being called
**Solution**: Check debug output - should show "Calling editor.graph.connect()..."

### Issue: "incompatible types" error

**Cause**: Trying to connect wrong port types
**Solution**: See docs/gui/PORT_TYPES_GUIDE.md for correct patterns

## Expected Workflow

For a complete AVTE training workflow, you need:

1. **AVTE2DLoader** node:
   - Outputs: `train_loader` (BATCH), `val_loader` (BATCH)

2. **UNet2D** node:
   - Outputs: `model` (MODEL)

3. **LossFunction** node:
   - Outputs: `loss_fn` (LOSS)

4. **Optimizer** node:
   - Inputs: `model` (MODEL)
   - Outputs: `optimizer` (OPTIMIZER)

5. **Trainer** node:
   - Inputs: `model` (MODEL), `dataloader` (BATCH), `val_dataloader` (BATCH), `loss_fn` (LOSS), `optimizer` (OPTIMIZER)

**Connections** (6 total):
1. AVTE2DLoader.train_loader → Trainer.dataloader
2. AVTE2DLoader.val_loader → Trainer.val_dataloader
3. UNet2D.model → Trainer.model
4. UNet2D.model → Optimizer.model
5. LossFunction.loss_fn → Trainer.loss_fn
6. Optimizer.optimizer → Trainer.optimizer

Each connection should produce debug output when created.

## Next Steps After Testing

Once you can confirm:
- ✅ Debug messages appear when creating connections
- ✅ "✓ Connected: ..." message appears
- ✅ "Graph now has N total links" shows increasing count
- ✅ Saved file contains connections in `"links": []` array
- ✅ Execution doesn't show "not connected" errors

Then the fix is working! You can remove the debug print statements if desired.
