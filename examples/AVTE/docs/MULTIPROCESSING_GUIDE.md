# Multiprocessing Guide

The preprocessing script now supports parallel processing using Python's multiprocessing module, significantly reducing preprocessing time.

## Performance Improvements

### Expected Speed-Up

| Workers | Processing Time | Speed-Up | Notes |
|---------|----------------|----------|-------|
| 1 (single) | 60 minutes | 1.0x | Baseline |
| 4 workers | ~15-20 minutes | 3-4x | Good for most systems |
| 8 workers | ~10-15 minutes | 4-6x | Requires 8+ CPU cores |
| 16 workers | ~8-12 minutes | 5-7x | Requires 16+ CPU cores |

**Note**: Actual speed-up depends on:
- Number of CPU cores
- Disk I/O speed (SSD vs HDD)
- Available RAM
- System load

## Usage

### Default (4 Workers)
```bash
python preprocess_2d_slices.py
```
Default is 4 workers, suitable for most systems.

### Custom Number of Workers
```bash
# Use 8 workers
python preprocess_2d_slices.py --num_workers 8

# Use all available CPUs
python preprocess_2d_slices.py --num_workers -1

# Single process (for debugging)
python preprocess_2d_slices.py --num_workers 0
```

### Recommended Settings

#### Desktop/Workstation (8+ cores)
```bash
python preprocess_2d_slices.py --num_workers 8
```

#### Server (16+ cores)
```bash
python preprocess_2d_slices.py --num_workers 16
```

#### Laptop (4 cores)
```bash
python preprocess_2d_slices.py --num_workers 2
```
Leave some cores for the system.

#### Debugging
```bash
python preprocess_2d_slices.py --num_workers 0
```
Single process mode makes error messages clearer.

## How It Works

### Architecture

1. **Main Process**:
   - Splits dataset into train/val/test
   - Creates worker pool
   - Distributes files to workers
   - Collects results and saves statistics

2. **Worker Processes**:
   - Each worker processes one NIfTI file at a time
   - Loads 3D volume
   - Extracts 2D slices with context
   - Saves individual .npz files
   - Returns slice count

3. **Progress Tracking**:
   - Uses tqdm with multiprocessing support
   - Shows real-time progress across all workers
   - Displays estimated time remaining

### File-Level Parallelism

The script parallelizes at the **file level**:
- Each NIfTI file (3D volume) is processed by one worker
- Slices within a file are processed sequentially
- Multiple files are processed in parallel

**Example with 100 cases and 4 workers:**
```
Worker 1: Processing case 1, 5, 9, 13, ...
Worker 2: Processing case 2, 6, 10, 14, ...
Worker 3: Processing case 3, 7, 11, 15, ...
Worker 4: Processing case 4, 8, 12, 16, ...
```

## Optimal Worker Count

### How to Choose

Use this formula as a starting point:
```
num_workers = max(1, CPU_cores - 1)
```

### Check Your CPU Count

```bash
# Linux/Mac
python3 -c "import multiprocessing; print(f'CPU cores: {multiprocessing.cpu_count()}')"

# Or use system commands
nproc  # Linux
sysctl -n hw.ncpu  # Mac
```

### Guidelines

1. **I/O Bound (slow disk)**:
   - Use fewer workers (2-4)
   - More workers won't help if disk is the bottleneck

2. **CPU Bound (fast SSD)**:
   - Use more workers (8-16)
   - Limited by CPU cores

3. **Memory Constrained**:
   - Reduce workers if running out of RAM
   - Each worker loads one full 3D volume
   - Estimate: ~2-4 GB RAM per worker

## Performance Tips

### 1. Use SSD Storage

```bash
# Check if your disk is SSD
lsblk -d -o name,rota
# rota=0 means SSD, rota=1 means HDD
```

**SSD**: Use more workers (8-16)
**HDD**: Use fewer workers (2-4)

### 2. Monitor System Resources

```bash
# Terminal 1: Run preprocessing
python preprocess_2d_slices.py --num_workers 8

# Terminal 2: Monitor resources
htop  # or top on Mac
```

Watch for:
- CPU usage (should be near 100% on multiple cores)
- Memory usage (should not hit swap)
- I/O wait (should be low)

### 3. Balance Workers with Other Tasks

If running other processes:
```bash
# Leave 2 cores free
python preprocess_2d_slices.py --num_workers $(($(nproc) - 2))
```

### 4. Disk I/O Optimization

For HDD (spinning disk):
```bash
# Use fewer workers to reduce disk thrashing
python preprocess_2d_slices.py --num_workers 2
```

For SSD:
```bash
# Use more workers to maximize throughput
python preprocess_2d_slices.py --num_workers -1
```

## Troubleshooting

### Memory Issues

**Symptom**: System becomes unresponsive, swap usage high

**Solution**:
```bash
# Reduce number of workers
python preprocess_2d_slices.py --num_workers 2

# Or use single process
python preprocess_2d_slices.py --num_workers 0
```

### Slow Performance Despite Multiple Workers

**Possible causes**:
1. **Disk bottleneck**: Check I/O wait with `htop`
2. **Memory pressure**: Check swap usage
3. **Other processes**: Close unnecessary applications

**Solution**:
```bash
# Check system load
uptime

# If load is high, reduce workers
python preprocess_2d_slices.py --num_workers 2
```

### Errors in Multiprocessing Mode

**Symptom**: Cryptic error messages, hard to debug

**Solution**:
```bash
# Use single process for clearer error messages
python preprocess_2d_slices.py --num_workers 0
```

Single process mode:
- Same functionality
- Easier debugging
- Clearer stack traces

### Process Hangs

**Symptom**: Progress bar stops updating

**Possible causes**:
1. Worker crashed silently
2. File system issue
3. Out of memory

**Solution**:
```bash
# Kill the process
Ctrl+C

# Try with single process to see error
python preprocess_2d_slices.py --num_workers 0
```

## Benchmarking

### Test Different Worker Counts

```bash
#!/bin/bash
# benchmark.sh - Test different worker counts

for workers in 1 2 4 8 16; do
    echo "Testing with $workers workers..."
    time python preprocess_2d_slices.py \
        --num_workers $workers \
        --output_dir /tmp/test_output_${workers}w
    echo "---"
done
```

Run the benchmark:
```bash
chmod +x benchmark.sh
./benchmark.sh
```

### Measure Time

```bash
# Using time command
time python preprocess_2d_slices.py --num_workers 8

# Output will show:
# real: Total wall-clock time
# user: CPU time spent in user mode
# sys:  CPU time spent in kernel mode
```

## Advanced Usage

### CPU Affinity (Linux only)

Bind process to specific CPUs:
```bash
taskset -c 0-7 python preprocess_2d_slices.py --num_workers 8
```

### Nice Priority

Run with lower priority to not interfere with other work:
```bash
nice -n 19 python preprocess_2d_slices.py --num_workers 8
```

### Background Processing

Run in background with nohup:
```bash
nohup python preprocess_2d_slices.py --num_workers 8 > preprocessing.log 2>&1 &

# Monitor progress
tail -f preprocessing.log
```

### Process Monitoring Script

```python
# monitor.py
import psutil
import time

while True:
    cpu = psutil.cpu_percent(interval=1, percpu=True)
    mem = psutil.virtual_memory().percent
    print(f"CPU: {cpu} | RAM: {mem}%")
    time.sleep(2)
```

## Implementation Details

### Worker Function

Each worker:
1. Receives file paths and configuration
2. Creates local preprocessor instance
3. Loads NIfTI file
4. Extracts and saves slices
5. Returns slice count

### Pool Management

- Uses `multiprocessing.Pool`
- `imap` for ordered processing with progress bar
- Automatic cleanup with context manager
- Exception handling in worker processes

### Progress Bar

- `tqdm` with multiprocessing support
- Updates in real-time across workers
- Shows ETA based on completed files

## Comparison: Single vs Multi-Process

### Single Process (--num_workers 0)

**Pros**:
- Simpler error messages
- Lower memory usage
- Better for debugging

**Cons**:
- Slower (uses only 1 CPU core)
- Not recommended for large datasets

**Use when**:
- Debugging issues
- Memory constrained
- Small datasets (<10 cases)

### Multi-Process (--num_workers > 0)

**Pros**:
- 3-7x faster
- Utilizes multiple CPU cores
- Recommended for production

**Cons**:
- Higher memory usage
- More complex error handling
- May overwhelm slow disks

**Use when**:
- Large datasets (50+ cases)
- Fast storage (SSD)
- Sufficient RAM available

## Example Commands

### Quick Test (Small Dataset)
```bash
python preprocess_2d_slices.py \
    --num_workers 2 \
    --train_ratio 0.9
```

### Production Run (Full Dataset)
```bash
python preprocess_2d_slices.py \
    --num_workers 8 \
    --window_size 2 \
    --padding_mode replicate
```

### Maximum Performance
```bash
python preprocess_2d_slices.py \
    --num_workers -1 \
    --output_dir /fast/ssd/path
```

### Debug Mode
```bash
python preprocess_2d_slices.py \
    --num_workers 0 \
    --train_ratio 0.95  # Process only a few test cases
```

## Summary

The multiprocessing support provides:
- **3-7x speed improvement** with proper configuration
- **Automatic CPU detection** with `--num_workers -1`
- **Fallback to single process** with `--num_workers 0`
- **Progress tracking** with tqdm
- **Error handling** per worker process

**Default of 4 workers** is a safe choice for most systems.

Adjust based on your hardware:
- More workers for faster preprocessing
- Fewer workers if memory constrained
- Single process for debugging

Happy preprocessing! 🚀
