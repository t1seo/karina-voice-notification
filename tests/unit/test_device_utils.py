"""Tests for device_utils module."""

import pytest


def test_detect_device_returns_device_info():
    """Test that detect_device returns a DeviceInfo object."""
    from scripts.device_utils import detect_device, DeviceInfo

    device = detect_device()
    assert isinstance(device, DeviceInfo)


def test_device_info_has_required_properties():
    """Test DeviceInfo has all required properties."""
    from scripts.device_utils import detect_device

    device = detect_device()
    assert hasattr(device, 'device_type')
    assert hasattr(device, 'device_name')
    assert hasattr(device, 'dtype')
    assert hasattr(device, 'torch_device')
    assert hasattr(device, 'is_gpu')
    assert hasattr(device, 'whisper_backend')


def test_torch_device_is_valid_string():
    """Test torch_device returns valid device string."""
    from scripts.device_utils import detect_device

    device = detect_device()
    assert device.torch_device in ('cuda:0', 'mps', 'cpu')
