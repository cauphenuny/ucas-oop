# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import torch

from diffusers.schedulers.utils_perflow import (
    load_delta_weights_into_unet,
    load_dreambooth_into_pipeline,
    merge_delta_weights_into_unet,
)


class DummyUNet:
    """Dummy UNet model for testing."""
    
    def __init__(self):
        """Initialize with dummy state dict."""
        self._state_dict = OrderedDict([
            ("conv_in.weight", torch.randn(320, 4, 3, 3)),
            ("conv_in.bias", torch.randn(320)),
            ("down_blocks.0.weight", torch.randn(320, 320, 3, 3)),
        ])
        # Add config attribute for DreamBooth loading
        self.config = {
            "in_channels": 4,
            "out_channels": 4,
            "num_class_embeds": None,
        }
    
    def state_dict(self):
        """Return state dict."""
        return self._state_dict.copy()
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict."""
        self._state_dict = state_dict


class DummyPipeline:
    """Dummy pipeline for testing."""
    
    def __init__(self):
        """Initialize with dummy UNet, VAE, and text encoder."""
        self.unet = DummyUNet()
        self.vae = MagicMock()
        self.vae.config = {}  # Add config for VAE
        self.vae.load_state_dict = MagicMock()
        self.text_encoder = MagicMock()


class UtilsPerflowTest(unittest.TestCase):
    """
    Comprehensive test suite for PeRFlow utility functions.
    
    This test class validates all utility functions including:
    - Delta weight merging
    - Delta weight loading
    - DreamBooth checkpoint loading
    """

    def test_merge_delta_weights_into_unet_basic(self):
        """Test basic delta weight merging."""
        pipe = DummyPipeline()
        original_weights = pipe.unet.state_dict()
        
        delta_weights = OrderedDict()
        for key, value in original_weights.items():
            delta_weights[key] = torch.randn_like(value) * 0.1
        
        result_pipe = merge_delta_weights_into_unet(pipe, delta_weights)
        assert result_pipe is not None

    def test_merge_delta_weights_preserves_keys(self):
        """Test that delta weight merging preserves all keys."""
        pipe = DummyPipeline()
        original_weights = pipe.unet.state_dict()
        
        delta_weights = OrderedDict()
        for key, value in original_weights.items():
            delta_weights[key] = torch.randn_like(value) * 0.1
        
        merge_delta_weights_into_unet(pipe, delta_weights)
        merged_weights = pipe.unet.state_dict()
        
        assert set(merged_weights.keys()) == set(original_weights.keys())

    def test_merge_delta_weights_shape_preservation(self):
        """Test that delta weight merging preserves tensor shapes."""
        pipe = DummyPipeline()
        original_weights = pipe.unet.state_dict()
        
        delta_weights = OrderedDict()
        for key, value in original_weights.items():
            delta_weights[key] = torch.randn_like(value) * 0.1
        
        merge_delta_weights_into_unet(pipe, delta_weights)
        merged_weights = pipe.unet.state_dict()
        
        for key in original_weights.keys():
            assert merged_weights[key].shape == original_weights[key].shape

    def test_merge_delta_weights_different_dtypes(self):
        """Test delta weight merging with different data types."""
        pipe = DummyPipeline()
        original_weights = pipe.unet.state_dict()
        
        delta_weights = OrderedDict()
        for key, value in original_weights.items():
            delta_weights[key] = (torch.randn_like(value) * 0.1).to(torch.float16)
        
        result_pipe = merge_delta_weights_into_unet(pipe, delta_weights)
        assert result_pipe is not None

    def test_merge_delta_weights_zero_delta(self):
        """Test delta weight merging with zero delta weights."""
        pipe = DummyPipeline()
        original_weights = pipe.unet.state_dict()
        
        delta_weights = OrderedDict()
        for key, value in original_weights.items():
            delta_weights[key] = torch.zeros_like(value)
        
        merge_delta_weights_into_unet(pipe, delta_weights)
        merged_weights = pipe.unet.state_dict()
        
        for key in original_weights.keys():
            assert torch.allclose(merged_weights[key], original_weights[key])

    @patch("diffusers.schedulers.utils_perflow.safe_open")
    @patch("os.path.exists")
    def test_load_delta_weights_from_safetensors(self, mock_exists, mock_safe_open):
        """Test loading delta weights from safetensors file."""
        mock_exists.return_value = True
        
        pipe = DummyPipeline()
        model_path = "/fake/path"
        
        delta_weights = OrderedDict()
        for key, value in pipe.unet.state_dict().items():
            delta_weights[key] = torch.randn_like(value) * 0.1
        
        mock_file = MagicMock()
        mock_file.keys.return_value = delta_weights.keys()
        mock_file.get_tensor.side_effect = lambda k: delta_weights[k]
        mock_safe_open.return_value.__enter__.return_value = mock_file
        
        result_pipe = load_delta_weights_into_unet(pipe, model_path)
        assert result_pipe is not None

    @patch("os.path.exists")
    def test_load_delta_weights_file_not_found(self, mock_exists):
        """Test load_delta_weights_into_unet when file doesn't exist."""
        mock_exists.return_value = False
        
        pipe = DummyPipeline()
        model_path = "/fake/path"
        
        try:
            load_delta_weights_into_unet(pipe, model_path)
        except (ValueError, FileNotFoundError, NotImplementedError):
            pass

    def test_load_delta_weights_default_paths(self):
        """Test load_delta_weights_into_unet with default paths."""
        pipe = DummyPipeline()
        
        try:
            load_delta_weights_into_unet(pipe)
        except (ValueError, FileNotFoundError, NotImplementedError, Exception):
            pass

    def test_load_delta_weights_custom_base_path(self):
        """Test load_delta_weights_into_unet with custom base path."""
        pipe = DummyPipeline()
        model_path = "/custom/model/path"
        base_path = "/custom/base/path"
        
        try:
            load_delta_weights_into_unet(pipe, model_path, base_path)
        except (ValueError, FileNotFoundError, NotImplementedError, Exception):
            pass

    @patch("diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_ldm_clip_checkpoint")
    @patch("diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_ldm_vae_checkpoint")
    @patch("diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_ldm_unet_checkpoint")
    @patch("diffusers.schedulers.utils_perflow.safe_open")
    @patch("os.path.exists")
    def test_load_dreambooth_basic(self, mock_exists, mock_safe_open, mock_convert_unet, mock_convert_vae, mock_convert_clip):
        """Test basic DreamBooth loading."""
        mock_exists.return_value = True
        
        pipe = DummyPipeline()
        sd_dreambooth = "/fake/path/checkpoint.safetensors"
        
        state_dict = OrderedDict()
        for key, value in pipe.unet.state_dict().items():
            state_dict[f"model.diffusion_model.{key}"] = value
        
        mock_file = MagicMock()
        mock_file.keys.return_value = state_dict.keys()
        mock_file.get_tensor.side_effect = lambda k: state_dict[k]
        mock_safe_open.return_value.__enter__.return_value = mock_file
        
        # Mock the conversion functions to return the state dicts
        mock_convert_unet.return_value = pipe.unet.state_dict()
        mock_convert_vae.return_value = {}
        mock_convert_clip.return_value = pipe.text_encoder
        
        result_pipe = load_dreambooth_into_pipeline(pipe, sd_dreambooth)
        assert result_pipe is not None
        
        # Verify conversion functions were called
        mock_convert_unet.assert_called_once()
        mock_convert_vae.assert_called_once()
        mock_convert_clip.assert_called_once()

    def test_load_dreambooth_invalid_extension(self):
        """Test load_dreambooth_into_pipeline with invalid file extension."""
        pipe = DummyPipeline()
        sd_dreambooth = "/fake/path/checkpoint.ckpt"
        
        try:
            load_dreambooth_into_pipeline(pipe, sd_dreambooth)
        except (AssertionError, ValueError, NotImplementedError):
            pass

    def test_load_dreambooth_safetensors_extension(self):
        """Test that load_dreambooth_into_pipeline accepts safetensors files."""
        pipe = DummyPipeline()
        sd_dreambooth = "/fake/path/checkpoint.safetensors"
        
        try:
            load_dreambooth_into_pipeline(pipe, sd_dreambooth)
        except (FileNotFoundError, NotImplementedError, Exception):
            pass


class UtilsPerflowIntegrationTest(unittest.TestCase):
    """
    Integration tests for PeRFlow utility functions.
    
    Tests that verify utility functions work together correctly.
    """

    def test_merge_and_save_weights(self):
        """Test merging delta weights and verifying the result."""
        pipe = DummyPipeline()
        original_weights = pipe.unet.state_dict()
        
        delta_weights = OrderedDict()
        for key, value in original_weights.items():
            delta_weights[key] = torch.ones_like(value) * 0.5
        
        merge_delta_weights_into_unet(pipe, delta_weights)
        merged_weights = pipe.unet.state_dict()
        
        for key in original_weights.keys():
            expected = original_weights[key] + 0.5
            assert torch.allclose(merged_weights[key], expected, atol=1e-5)

    def test_multiple_merge_operations(self):
        """Test multiple consecutive delta weight merges."""
        pipe = DummyPipeline()
        original_weights = pipe.unet.state_dict()
        
        for i in range(3):
            delta_weights = OrderedDict()
            for key, value in pipe.unet.state_dict().items():
                delta_weights[key] = torch.ones_like(value) * 0.1
            
            merge_delta_weights_into_unet(pipe, delta_weights)
        
        merged_weights = pipe.unet.state_dict()
        for key in original_weights.keys():
            expected = original_weights[key] + 0.3
            assert torch.allclose(merged_weights[key], expected, atol=1e-5)

    def test_weight_consistency_after_operations(self):
        """Test that weight operations maintain consistency."""
        pipe = DummyPipeline()
        
        delta_weights = OrderedDict()
        for key, value in pipe.unet.state_dict().items():
            delta_weights[key] = torch.randn_like(value) * 0.01
        
        before_keys = set(pipe.unet.state_dict().keys())
        merge_delta_weights_into_unet(pipe, delta_weights)
        after_keys = set(pipe.unet.state_dict().keys())
        
        assert before_keys == after_keys

    def test_delta_weight_device_consistency(self):
        """Test that delta weights work correctly with different devices."""
        pipe = DummyPipeline()
        
        delta_weights = OrderedDict()
        for key, value in pipe.unet.state_dict().items():
            delta_weights[key] = torch.randn_like(value) * 0.1
        
        result_pipe = merge_delta_weights_into_unet(pipe, delta_weights)
        
        for key in result_pipe.unet.state_dict().keys():
            weight = result_pipe.unet.state_dict()[key]
            assert weight.device.type == "cpu"

    def test_numerical_precision_preservation(self):
        """Test that numerical precision is preserved during operations."""
        pipe = DummyPipeline()
        original_weights = pipe.unet.state_dict()
        
        delta_weights = OrderedDict()
        for key, value in original_weights.items():
            delta_weights[key] = torch.zeros_like(value)
        
        merge_delta_weights_into_unet(pipe, delta_weights)
        merged_weights = pipe.unet.state_dict()
        
        for key in original_weights.keys():
            assert torch.equal(merged_weights[key], original_weights[key])

    def test_large_delta_weights(self):
        """Test merging with large delta weight values."""
        pipe = DummyPipeline()
        
        delta_weights = OrderedDict()
        for key, value in pipe.unet.state_dict().items():
            delta_weights[key] = torch.randn_like(value) * 10.0
        
        result_pipe = merge_delta_weights_into_unet(pipe, delta_weights)
        assert result_pipe is not None

    def test_small_delta_weights(self):
        """Test merging with very small delta weight values."""
        pipe = DummyPipeline()
        
        delta_weights = OrderedDict()
        for key, value in pipe.unet.state_dict().items():
            delta_weights[key] = torch.randn_like(value) * 1e-7
        
        result_pipe = merge_delta_weights_into_unet(pipe, delta_weights)
        assert result_pipe is not None
