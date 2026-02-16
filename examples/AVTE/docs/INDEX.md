# AVTE 2D Segmentation Documentation

## Quick Start

- **[README.md](README.md)** - Main documentation and overview
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Quick start guide for new users
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command reference and cheat sheet

## Guides

- **[MULTIPROCESSING_GUIDE.md](MULTIPROCESSING_GUIDE.md)** - Performance optimization with multiprocessing
- **[GUI_NODE_GUIDE.md](GUI_NODE_GUIDE.md)** - Using the AVTE2DLoader node in the visual pipeline editor
- **[PATH_CONFIGURATION.md](PATH_CONFIGURATION.md)** - Setting up paths for your environment

## Reference

- **[SPLITTING_CHANGES.md](SPLITTING_CHANGES.md)** - Migration guide for v1.2.0 splitting changes
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes
- **[NODE_INTEGRATION_SUMMARY.md](NODE_INTEGRATION_SUMMARY.md)** - Technical summary of GUI node integration

## Documentation Structure

### For New Users
1. Start with [README.md](README.md) for overview
2. Follow [GETTING_STARTED.md](GETTING_STARTED.md) for setup
3. Use [QUICK_REFERENCE.md](QUICK_REFERENCE.md) as reference

### For GUI Users
1. Read [GUI_NODE_GUIDE.md](GUI_NODE_GUIDE.md) for node usage
2. See [NODE_INTEGRATION_SUMMARY.md](NODE_INTEGRATION_SUMMARY.md) for technical details

### For Advanced Users
1. Check [MULTIPROCESSING_GUIDE.md](MULTIPROCESSING_GUIDE.md) for performance tuning
2. Read [SPLITTING_CHANGES.md](SPLITTING_CHANGES.md) if migrating from v1.1.0
3. Review [CHANGELOG.md](CHANGELOG.md) for version history

## Quick Links

### Commands
```bash
# Preprocessing
python preprocess_2d_slices.py --num_workers 8

# Test dataloader
python avte_dataloader.py

# Test GUI node
python avte_dataloader_node.py

# Example usage
python example_usage.py
```

### Python API
```python
from avte_dataloader import create_avte_dataloaders

train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir='/path/to/2D_data',
    batch_size=16,
    train_ratio=0.8,
    val_ratio=0.1
)
```

### GUI Node
```python
import sys
sys.path.insert(0, '/path/to/AVTE')
import avte_dataloader_node
# Node auto-registers as 'AVTE2DLoader' in 'data' category
```

## File Descriptions

| File | Size | Purpose |
|------|------|---------|
| README.md | 10KB | Main documentation with examples |
| GETTING_STARTED.md | 8.4KB | Step-by-step setup guide |
| QUICK_REFERENCE.md | 4.6KB | Command and API reference |
| MULTIPROCESSING_GUIDE.md | 8.8KB | Performance optimization guide |
| GUI_NODE_GUIDE.md | 12KB | Visual pipeline editor node guide |
| PATH_CONFIGURATION.md | 3.7KB | Path setup instructions |
| SPLITTING_CHANGES.md | 7.6KB | v1.2.0 migration guide |
| CHANGELOG.md | 7.6KB | Version history |
| NODE_INTEGRATION_SUMMARY.md | 7.7KB | Technical integration summary |

## Version Information

- **Current Version**: 1.2.0
- **Release Date**: 2026-02-08
- **Module**: AVTE 2D Segmentation
- **Framework**: Medical Imaging Framework

## Support

For issues, questions, or contributions:
- Check the relevant documentation file above
- Review [CHANGELOG.md](CHANGELOG.md) for recent changes
- See example scripts in the parent directory

---

**Last Updated**: 2026-02-08
