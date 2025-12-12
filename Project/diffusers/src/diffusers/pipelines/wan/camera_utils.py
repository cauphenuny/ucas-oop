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
Camera control utilities for Wan video generation pipelines.

This module provides functions to process camera trajectories from txt files
(VideoX-Fun/CameraCtrl format) and convert them to Plücker ray embeddings
for controlling camera motion in video generation.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from einops import rearrange
from packaging import version as pver


class Camera:
    """
    Camera intrinsic and extrinsic parameters.
    
    Modified from: https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    
    Args:
        entry: Camera parameter list containing [frame_id, fx, fy, cx, cy, _, _, r11, r12, ..., t3]
               where the rotation matrix (3x3) and translation vector (3x1) form the world-to-camera matrix.
    """

    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def custom_meshgrid(*args):
    """
    Custom meshgrid function that handles different PyTorch versions.
    
    Modified from: https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    
    Returns appropriate meshgrid based on PyTorch version.
    """
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def get_relative_pose(cam_params):
    """
    Convert absolute camera poses to relative poses.
    
    Modified from: https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    
    Args:
        cam_params: List of Camera objects with w2c_mat and c2w_mat attributes
        
    Returns:
        np.ndarray: Relative camera poses as numpy array of shape [num_frames, 4, 4]
    """
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([[1, 0, 0, 0], [0, 1, 0, -cam_to_origin], [0, 0, 1, 0], [0, 0, 0, 1]])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [
        target_cam_c2w,
    ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def ray_condition(K, c2w, H, W, device):
    """
    Generate Plücker ray embeddings from camera intrinsics and extrinsics.
    
    Modified from: https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    
    The Plücker coordinates represent oriented lines in 3D space using 6 parameters:
    - Direction vector (3 params)
    - Moment vector (3 params, cross product of position and direction)
    
    Args:
        K: Camera intrinsics tensor of shape [B, V, 4] containing [fx, fy, cx, cy]
        c2w: Camera-to-world transformation matrices of shape [B, V, 4, 4]
        H: Height of the output embedding
        W: Width of the output embedding
        device: PyTorch device to create tensors on
        
    Returns:
        torch.Tensor: Plücker embeddings of shape [B, V, H, W, 6]
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ directions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    return plucker


def process_camera_txt(
    txt_path: Union[str, Path],
    width: int = 672,
    height: int = 384,
    original_pose_width: int = 1280,
    original_pose_height: int = 720,
    device: Union[str, torch.device] = "cpu",
    num_frames: Optional[int] = None,
    fix_frame_id: bool = True,
) -> torch.Tensor:
    """
    Process camera trajectory from a txt file and generate Plücker embeddings.
    
    This function reads camera parameters from a txt file in CameraCtrl/VideoX-Fun format
    and generates Plücker ray embeddings suitable for camera-controlled video generation.
    
    The txt file format has one header line followed by data lines with format:
    frame_id fx fy cx cy _ _ r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3
    
    Where:
    - frame_id: Frame index (will be fixed to sequential if fix_frame_id=True)
    - fx, fy, cx, cy: Camera intrinsic parameters
    - r11-r33, t1-t3: 3x4 world-to-camera transformation matrix [R|t]
    
    Args:
        txt_path: Path to the camera trajectory txt file
        width: Target video width
        height: Target video height
        original_pose_width: Original width used when generating the poses
        original_pose_height: Original height used when generating the poses
        device: Device to create tensors on
        num_frames: If specified, clip or interpolate to this number of frames
        fix_frame_id: If True, replace frame_id column with sequential indices (0, 1, 2, ...)
                     to fix the VideoX-Fun issue where frame_id is always 0
        
    Returns:
        torch.Tensor: Plücker embeddings of shape [num_frames, height, width, 6]
        
    Raises:
        FileNotFoundError: If txt_path doesn't exist
        ValueError: If txt file has invalid format
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Camera txt file not found: {txt_path}")

    with open(txt_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"Camera txt file must have at least 2 lines (header + data), got {len(lines)}")

    # Parse camera parameters (skip header line)
    poses = [line.strip().split() for line in lines[1:]]
    cam_params = []
    for frame_idx, pose in enumerate(poses):
        if len(pose) != 19:
            raise ValueError(f"Each line must have 19 values, line {frame_idx + 2} has {len(pose)}")
        
        params = [float(x) for x in pose]
        
        # Fix frame_id if requested (addresses VideoX-Fun issue)
        if fix_frame_id:
            params[0] = float(frame_idx)
        
        cam_params.append(params)

    # Clip or interpolate to target number of frames if specified
    if num_frames is not None and len(cam_params) != num_frames:
        if len(cam_params) > num_frames:
            # Clip to num_frames
            cam_params = cam_params[:num_frames]
        else:
            # Simple interpolation: repeat last frame
            while len(cam_params) < num_frames:
                cam_params.append(cam_params[-1])

    # Create Camera objects
    cam_objects = [Camera(cam_param) for cam_param in cam_params]

    # Adjust intrinsics based on aspect ratios
    sample_wh_ratio = width / height
    pose_wh_ratio = original_pose_width / original_pose_height

    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = height * pose_wh_ratio
        for cam_param in cam_objects:
            cam_param.fx = resized_ori_w * cam_param.fx / width
    else:
        resized_ori_h = width / pose_wh_ratio
        for cam_param in cam_objects:
            cam_param.fy = resized_ori_h * cam_param.fy / height

    # Build intrinsic matrix
    intrinsic = np.asarray(
        [[cam_param.fx * width, cam_param.fy * height, cam_param.cx * width, cam_param.cy * height]
         for cam_param in cam_objects],
        dtype=np.float32,
    )

    K = torch.as_tensor(intrinsic)[None]  # [1, num_frames, 4]
    c2ws = get_relative_pose(cam_objects)
    c2ws = torch.as_tensor(c2ws)[None]  # [1, num_frames, 4, 4]
    
    # Generate Plücker embeddings
    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(
        0, 3, 1, 2
    ).contiguous()  # num_frames, 6, H, W
    plucker_embedding = plucker_embedding[None]  # 1, num_frames, 6, H, W
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]  # num_frames, H, W, 6

    return plucker_embedding


def process_camera_params(
    cam_params,
    width: int = 672,
    height: int = 384,
    original_pose_width: int = 1280,
    original_pose_height: int = 720,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Process camera parameters directly and generate Plücker embeddings.
    
    This is a lower-level function for when you already have parsed camera parameters
    rather than reading from a file.
    
    Args:
        cam_params: List of camera parameter lists, each with 19 values
        width: Target video width
        height: Target video height
        original_pose_width: Original width used when generating the poses
        original_pose_height: Original height used when generating the poses
        device: Device to create tensors on
        
    Returns:
        torch.Tensor: Plücker embeddings of shape [num_frames, height, width, 6]
    """
    cam_objects = [Camera(cam_param) for cam_param in cam_params]

    sample_wh_ratio = width / height
    pose_wh_ratio = original_pose_width / original_pose_height

    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = height * pose_wh_ratio
        for cam_param in cam_objects:
            cam_param.fx = resized_ori_w * cam_param.fx / width
    else:
        resized_ori_h = width / pose_wh_ratio
        for cam_param in cam_objects:
            cam_param.fy = resized_ori_h * cam_param.fy / height

    intrinsic = np.asarray(
        [[cam_param.fx * width, cam_param.fy * height, cam_param.cx * width, cam_param.cy * height]
         for cam_param in cam_objects],
        dtype=np.float32,
    )

    K = torch.as_tensor(intrinsic)[None]  # [1, num_frames, 4]
    c2ws = get_relative_pose(cam_objects)
    c2ws = torch.as_tensor(c2ws)[None]  # [1, num_frames, 4, 4]
    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(
        0, 3, 1, 2
    ).contiguous()  # num_frames, 6, H, W
    plucker_embedding = plucker_embedding[None]
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
    return plucker_embedding
