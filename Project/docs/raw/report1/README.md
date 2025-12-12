# Documentation

This directory contains comprehensive documentation for the ucas-oop project.

## Available Documents

### Wan Camera Control Implementation Report

**File:** [wan-camera-control-implementation-report.md](wan-camera-control-implementation-report.md)

A detailed technical report documenting the analysis and implementation of camera control functionality for the Wan video generation pipeline in HuggingFace Diffusers.

**Contents:**
- Executive summary and achievements
- Background and motivation
- Analysis of VideoX-Fun and DiffSynth-Studio implementations
- Implementation details and architecture
- Code structure and file organization
- Testing and validation procedures
- Usage examples and tutorials
- Future work and roadmap
- Technical specifications and references

**Key Features Documented:**
- Camera utilities module (`camera_utils.py`)
- VideoX-Fun txt format support
- Frame ID bug fix implementation
- Sample camera trajectories (12 files)
- Comprehensive testing infrastructure
- Integration with Diffusers pipelines

## Quick Links

### Implementation Files

**Core Module:**
- `Project/diffusers/src/diffusers/pipelines/wan/camera_utils.py` - Camera control utilities

**Tests:**
- `Project/diffusers/tests/pipelines/wan/test_wan_camera_utils.py` - Unit tests

**Examples:**
- `Project/diffusers/examples/community/wan_camera_control_example.py` - Example script
- `Project/diffusers/examples/community/wan_camera_samples/*.txt` - Sample trajectories

**Documentation:**
- `Project/diffusers/docs/source/en/api/pipelines/wan.md` - API documentation

### Usage

To use camera control in your project:

```python
from diffusers.pipelines.wan import process_camera_txt

# Load and process camera trajectory
camera_embeddings = process_camera_txt(
    txt_path="path/to/trajectory.txt",
    width=672,
    height=384,
    fix_frame_id=True,  # Fix VideoX-Fun frame_id bug
)

# Use embeddings for camera-controlled video generation
# Output shape: [num_frames, height, width, 6]
```

### Sample Trajectories

Available in `Project/diffusers/examples/community/wan_camera_samples/`:
- Zoom_In.txt / Zoom_Out.txt
- Pan_Left.txt / Pan_Right.txt
- Pan_Up.txt / Pan_Down.txt
- Pan_Left_Up.txt / Pan_Left_Down.txt / Pan_Right_Up.txt / Pan_Right_Down.txt
- CW.txt / ACW.txt

## Project Structure

```
ucas-oop/
├── docs/                                    # This directory
│   ├── README.md                           # This file
│   └── wan-camera-control-implementation-report.md
├── Project/
│   └── diffusers/
│       ├── src/diffusers/pipelines/wan/
│       │   ├── camera_utils.py            # Core implementation
│       │   └── __init__.py                # Exports
│       ├── tests/pipelines/wan/
│       │   └── test_wan_camera_utils.py   # Tests
│       ├── examples/community/
│       │   ├── wan_camera_control_example.py
│       │   └── wan_camera_samples/*.txt   # 12 sample files
│       └── docs/source/en/api/pipelines/
│           └── wan.md                      # API docs (updated)
└── Assignment1/                            # Other course materials
```

## Contributing

When adding new documentation:

1. Place comprehensive technical reports in this `docs/` directory
2. Add brief summaries to this README
3. Keep code-specific documentation with the code (docstrings, inline comments)
4. Update relevant API documentation in the project

## Version History

- **v1.0** (December 12, 2025) - Initial camera control implementation report
  - Complete camera utilities module
  - VideoX-Fun format support
  - Frame ID fix implementation
  - 12 sample trajectory files
  - Comprehensive testing and examples

## License

This documentation follows the same license as the main project. The camera control implementation is based on:
- VideoX-Fun (Apache 2.0)
- CameraCtrl (compatible license)
- DiffSynth-Studio (Apache 2.0)

All modifications and additions are compatible with HuggingFace Diffusers (Apache 2.0).
