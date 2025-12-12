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

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from diffusers.pipelines.wan.camera_utils import (
    Camera,
    custom_meshgrid,
    get_relative_pose,
    process_camera_params,
    process_camera_txt,
    ray_condition,
)


class CameraUtilsTest(unittest.TestCase):
    def test_camera_class(self):
        """Test Camera class initialization"""
        # Create a sample camera entry with 19 values
        # [frame_id, fx, fy, cx, cy, _, _, r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3]
        entry = [0, 0.5, 0.9, 0.5, 0.5, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        camera = Camera(entry)
        
        self.assertAlmostEqual(camera.fx, 0.5)
        self.assertAlmostEqual(camera.fy, 0.9)
        self.assertAlmostEqual(camera.cx, 0.5)
        self.assertAlmostEqual(camera.cy, 0.5)
        self.assertEqual(camera.w2c_mat.shape, (4, 4))
        self.assertEqual(camera.c2w_mat.shape, (4, 4))
        
        # Test that c2w is inverse of w2c
        identity = camera.w2c_mat @ camera.c2w_mat
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-6)

    def test_custom_meshgrid(self):
        """Test custom_meshgrid function"""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0])
        
        grid_x, grid_y = custom_meshgrid(x, y)
        
        self.assertEqual(grid_x.shape, (3, 2))
        self.assertEqual(grid_y.shape, (3, 2))

    def test_get_relative_pose(self):
        """Test get_relative_pose function"""
        # Create two simple cameras
        entry1 = [0, 0.5, 0.9, 0.5, 0.5, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        entry2 = [1, 0.5, 0.9, 0.5, 0.5, 0, 0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        cameras = [Camera(entry1), Camera(entry2)]
        
        relative_poses = get_relative_pose(cameras)
        
        self.assertEqual(relative_poses.shape, (2, 4, 4))
        self.assertEqual(relative_poses.dtype, np.float32)

    def test_ray_condition(self):
        """Test ray_condition function"""
        B, V, H, W = 1, 2, 8, 8
        device = "cpu"
        
        # Create sample intrinsics and extrinsics
        K = torch.randn(B, V, 4)  # [fx, fy, cx, cy]
        c2w = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, V, 4, 4).clone()
        
        plucker = ray_condition(K, c2w, H, W, device)
        
        self.assertEqual(plucker.shape, (B, V, H, W, 6))
        self.assertEqual(plucker.dtype, c2w.dtype)

    def test_process_camera_txt(self):
        """Test process_camera_txt function with a temporary txt file"""
        # Create a temporary txt file with camera data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Write header
            f.write("header line\n")
            # Write camera data (19 values per line)
            for i in range(5):
                # frame_id fx fy cx cy _ _ r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3
                line = f"0 0.532 0.946 0.5 0.5 0 0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 -{i * 0.1}\n"
                f.write(line)
            temp_path = f.name
        
        try:
            # Test basic functionality
            embedding = process_camera_txt(temp_path, width=64, height=64)
            
            self.assertEqual(embedding.shape, (5, 64, 64, 6))
            self.assertEqual(embedding.dtype, torch.float32)
            
            # Test with num_frames clipping
            embedding_clipped = process_camera_txt(temp_path, width=64, height=64, num_frames=3)
            self.assertEqual(embedding_clipped.shape, (3, 64, 64, 6))
            
            # Test with num_frames expansion
            embedding_expanded = process_camera_txt(temp_path, width=64, height=64, num_frames=10)
            self.assertEqual(embedding_expanded.shape, (10, 64, 64, 6))
            
            # Test fix_frame_id functionality
            embedding_fixed = process_camera_txt(temp_path, width=64, height=64, fix_frame_id=True)
            self.assertEqual(embedding_fixed.shape, (5, 64, 64, 6))
            
        finally:
            # Clean up temp file
            Path(temp_path).unlink()

    def test_process_camera_txt_invalid_file(self):
        """Test process_camera_txt with invalid file"""
        with self.assertRaises(FileNotFoundError):
            process_camera_txt("/nonexistent/path.txt")

    def test_process_camera_params(self):
        """Test process_camera_params function"""
        # Create sample camera parameters
        cam_params = []
        for i in range(3):
            params = [i, 0.5, 0.9, 0.5, 0.5, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -i * 0.1]
            cam_params.append(params)
        
        embedding = process_camera_params(cam_params, width=64, height=64)
        
        self.assertEqual(embedding.shape, (3, 64, 64, 6))
        self.assertEqual(embedding.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
