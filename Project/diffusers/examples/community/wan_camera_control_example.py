#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script demonstrating camera control for Wan video generation.

This example shows how to use camera trajectory txt files to control
camera motion in video generation using the Wan pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import torch
    from diffusers.pipelines.wan.camera_utils import process_camera_txt
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease install the required dependencies:")
    print("  pip install torch numpy einops packaging")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test camera control utilities for Wan video generation")
    parser.add_argument(
        "--camera_txt",
        type=str,
        default="wan_camera_samples/Zoom_In.txt",
        help="Path to camera trajectory txt file",
    )
    parser.add_argument("--width", type=int, default=672, help="Video width")
    parser.add_argument("--height", type=int, default=384, help="Video height")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames (optional)")
    parser.add_argument(
        "--no_fix_frame_id",
        action="store_true",
        help="Don't fix frame_id (keep original values from txt)",
    )

    args = parser.parse_args()

    # Resolve path relative to script location
    script_dir = Path(__file__).parent
    camera_txt_path = script_dir / args.camera_txt

    if not camera_txt_path.exists():
        print(f"Error: Camera txt file not found: {camera_txt_path}")
        print(f"\nAvailable sample files in wan_camera_samples/:")
        samples_dir = script_dir / "wan_camera_samples"
        if samples_dir.exists():
            for txt_file in sorted(samples_dir.glob("*.txt")):
                print(f"  - {txt_file.name}")
        sys.exit(1)

    print(f"Processing camera trajectory from: {camera_txt_path}")
    print(f"Target resolution: {args.width}x{args.height}")
    print(f"Fix frame_id: {not args.no_fix_frame_id}")

    try:
        # Process the camera txt file
        embeddings = process_camera_txt(
            camera_txt_path,
            width=args.width,
            height=args.height,
            device="cpu",
            num_frames=args.num_frames,
            fix_frame_id=not args.no_fix_frame_id,
        )

        print(f"\n✓ Successfully processed camera trajectory!")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Number of frames: {embeddings.shape[0]}")
        print(f"  Resolution: {embeddings.shape[1]}x{embeddings.shape[2]}")
        print(f"  Channels (Plücker coords): {embeddings.shape[3]}")
        print(f"  Data type: {embeddings.dtype}")
        print(f"  Device: {embeddings.device}")

        # Show some statistics
        print(f"\n  Embedding statistics:")
        print(f"    Min value: {embeddings.min().item():.6f}")
        print(f"    Max value: {embeddings.max().item():.6f}")
        print(f"    Mean value: {embeddings.mean().item():.6f}")
        print(f"    Std value: {embeddings.std().item():.6f}")

        print(f"\n✓ Camera control utilities are working correctly!")
        print(f"\nThese embeddings can be used as camera control input for Wan video generation.")

    except Exception as e:
        print(f"\n✗ Error processing camera trajectory: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
