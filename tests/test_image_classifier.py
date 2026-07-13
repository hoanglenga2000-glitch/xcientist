"""Tests for the image-classifier config layer (torch-free surface).

Training/inference need torch and run on the GPU server, so these tests cover the
pure logic: shape inference, architecture selection, and config validation.
"""
from __future__ import annotations

import json

import pytest

from research_agent_workstation.server.training.image_classifier import (
    ArchConfig,
    ImageClassifier,
    build_arch_config,
    infer_image_shape,
)


def test_infer_shape_mnist_784():
    assert infer_image_shape(784) == (1, 28, 28)


def test_infer_shape_cifar_3072_rgb():
    # 3072 = 3 * 32 * 32
    assert infer_image_shape(3072, channels=3) == (3, 32, 32)


def test_infer_shape_1024_grayscale_32():
    assert infer_image_shape(1024) == (1, 32, 32)


def test_infer_shape_fallback_nonsquare():
    # 800 isn't a clean common-edge square; fallback returns a near-square edge.
    shape = infer_image_shape(800)
    assert shape[0] == 1
    assert shape[1] == shape[2]  # square fallback


def test_build_arch_small_image_is_cnn():
    cfg = build_arch_config(784, 10)
    assert cfg.model_type == "cnn"
    assert cfg.input_shape == (1, 28, 28)
    assert cfg.num_classes == 10


def test_build_arch_medium_image_is_resnet():
    # 64x64 grayscale = 4096 pixels
    cfg = build_arch_config(4096, 5)
    assert cfg.model_type == "resnet18"
    assert cfg.input_shape == (1, 64, 64)


def test_build_arch_large_image_is_vit():
    # 224x224 = 50176
    cfg = build_arch_config(50176, 1000)
    assert cfg.model_type == "vit_tiny"


def test_build_arch_prefer_override():
    cfg = build_arch_config(784, 10, prefer="resnet18")
    assert cfg.model_type == "resnet18"


def test_arch_config_json_serializable():
    cfg = build_arch_config(784, 10)
    json.dumps(cfg.to_dict())


def test_classifier_validates_num_classes():
    with pytest.raises(ValueError):
        ImageClassifier(ArchConfig(model_type="cnn", input_shape=(1, 28, 28), num_classes=1))


def test_classifier_validates_input_shape():
    with pytest.raises(ValueError):
        ImageClassifier(ArchConfig(model_type="cnn", input_shape=(1, 0, 28), num_classes=10))


def test_classifier_accepts_valid_config():
    clf = ImageClassifier(build_arch_config(784, 10))
    assert clf.config.num_classes == 10


def test_fit_without_torch_raises_clear_error():
    clf = ImageClassifier(build_arch_config(784, 10))
    with pytest.raises(RuntimeError, match="GPU"):
        clf.fit(None, None)
