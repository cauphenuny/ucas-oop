# Camera Control Quick Reference

**Last Updated:** December 12, 2025  
**Status:** Production Ready ✅

## 🚀 Quick Start (30 seconds)

```python
from diffusers.pipelines.wan import process_camera_txt

# Load camera trajectory
embeddings = process_camera_txt(
    "wan_camera_samples/Zoom_In.txt",
    width=672,
    height=384,
    fix_frame_id=True,
)
# Output: [81, 384, 672, 6] Plücker embeddings
```

## 📦 Installation

```bash
cd Project/diffusers
pip install -e .
```

## 📁 File Locations

| What | Where |
|------|-------|
| Core Module | `Project/diffusers/src/diffusers/pipelines/wan/camera_utils.py` |
| Tests | `Project/diffusers/tests/pipelines/wan/test_wan_camera_utils.py` |
| Example | `Project/diffusers/examples/community/wan_camera_control_example.py` |
| Samples | `Project/diffusers/examples/community/wan_camera_samples/*.txt` |
| Full Docs | `docs/wan-camera-control-implementation-report.md` |

## 🎥 Available Camera Movements

```bash
# Run any sample trajectory
cd Project/diffusers/examples/community
python wan_camera_control_example.py --camera_txt wan_camera_samples/[FILE]
```

| File | Movement | Description |
|------|----------|-------------|
| `Zoom_In.txt` | Forward zoom | Camera moves closer |
| `Zoom_Out.txt` | Backward zoom | Camera moves away |
| `Pan_Left.txt` | Left pan | Horizontal left |
| `Pan_Right.txt` | Right pan | Horizontal right |
| `Pan_Up.txt` | Up pan | Vertical up |
| `Pan_Down.txt` | Down pan | Vertical down |
| `Pan_Left_Up.txt` | Diagonal | Up-left movement |
| `Pan_Left_Down.txt` | Diagonal | Down-left movement |
| `Pan_Right_Up.txt` | Diagonal | Up-right movement |
| `Pan_Right_Down.txt` | Diagonal | Down-right movement |
| `CW.txt` | Rotate | Clockwise rotation |
| `ACW.txt` | Rotate | Anti-clockwise rotation |

## 🔧 Common Use Cases

### 1. Load Trajectory

```python
from diffusers.pipelines.wan import process_camera_txt

embeddings = process_camera_txt("path/to/camera.txt", width=672, height=384)
```

### 2. Clip Frames

```python
# Get only first 50 frames
embeddings = process_camera_txt("camera.txt", width=672, height=384, num_frames=50)
```

### 3. GPU Processing

```python
embeddings = process_camera_txt("camera.txt", width=672, height=384, device="cuda")
```

### 4. Custom Parameters

```python
from diffusers.pipelines.wan import process_camera_params

params = [[0, 0.5, 0.9, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]]
embeddings = process_camera_params(params, width=672, height=384)
```

## 📝 Txt File Format

```
header_line_ignored
frame_id fx fy cx cy _ _ r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3
[repeat for each frame]
```

**19 values per line:**
- `frame_id`: Frame index (0, 1, 2, ...)
- `fx, fy`: Focal lengths (normalized)
- `cx, cy`: Principal point (normalized)
- `_, _`: Placeholders (ignored)
- `r11-r33, t1-t3`: 3×4 camera matrix [R|t]

## 🐛 Troubleshooting

### Import Error
```bash
# Solution: Install in editable mode
cd Project/diffusers && pip install -e .
```

### Invalid Txt File
```python
# Problem: Wrong number of values
# Solution: Check format - must have 19 values per line
```

### Memory Issues
```python
# Solution: Use lower resolution
embeddings = process_camera_txt(txt_path, width=512, height=384)
```

## 🔑 Key Features

- ✅ **VideoX-Fun Compatible**: Use existing txt files
- ✅ **Frame ID Fix**: Automatically fixes VideoX-Fun bug
- ✅ **Flexible Sizing**: Clip or extend to any frame count
- ✅ **Aspect Ratio**: Automatic intrinsic adjustment
- ✅ **GPU Support**: Process on CUDA if available

## 📊 API Functions

### Main Functions

```python
# Load from txt file
process_camera_txt(txt_path, width, height, ...)
# → [num_frames, H, W, 6]

# Process parameters directly
process_camera_params(cam_params, width, height, ...)
# → [num_frames, H, W, 6]
```

### Low-Level Functions

```python
# Camera class
Camera(entry)  # Create camera from parameters

# Geometric operations
get_relative_pose(cam_params)  # Relative poses
ray_condition(K, c2w, H, W, device)  # Plücker embeddings
custom_meshgrid(*args)  # Version-safe meshgrid
```

## 📚 Documentation

| Document | Size | Purpose |
|----------|------|---------|
| [Quick Reference](QUICK_REFERENCE.md) | 3KB | This file - fast lookup |
| [Implementation Summary](IMPLEMENTATION_SUMMARY.md) | 9KB | Overview and quick start |
| [Technical Report](wan-camera-control-implementation-report.md) | 24KB | Complete documentation |
| [API Docs](../Project/diffusers/docs/source/en/api/pipelines/wan.md) | - | Official API reference |

## 🧪 Testing

```bash
# Run unit tests
cd Project/diffusers
python -m pytest tests/pipelines/wan/test_wan_camera_utils.py -v

# Or directly
python tests/pipelines/wan/test_wan_camera_utils.py
```

## 🎯 Next Steps

1. **Try Examples**: Run `wan_camera_control_example.py`
2. **Read Docs**: Check [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
3. **Create Custom**: Make your own camera trajectory txt
4. **Integrate**: Use in your video generation pipeline

## 📞 Support

- **Full Documentation**: [wan-camera-control-implementation-report.md](wan-camera-control-implementation-report.md)
- **Repository**: https://github.com/cauphenuny/ucas-oop
- **Branch**: copilot/analyze-wan-video-pipeline

---

**Pro Tip:** Start with the example script to verify everything works:
```bash
cd Project/diffusers/examples/community
python wan_camera_control_example.py
```
