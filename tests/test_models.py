"""Unit tests for coronet.models architectures using random tensors.

These tests only check forward-pass shapes and dtypes with randomly
initialized weights and random inputs — they verify the architectures
are wired correctly, not that they are trained/accurate. They require
torch, torchvision, and monai to be installed (see requirements.txt).
"""

from __future__ import annotations

import pytest
import torch

from coronet.exceptions import ConfigurationError
from coronet.models.architectures import CustomCNN3D, FeatureExtractorFactory, VesselResNet3D
from coronet.models.sequence_models import PlaqueSequenceModel


def test_custom_cnn3d_output_shape() -> None:
    model = CustomCNN3D(output_dim=512)
    x = torch.randn(2, 1, 32, 32, 32)
    out = model(x)
    assert out.shape == (2, 512)


def test_vessel_resnet3d_output_shape() -> None:
    model = VesselResNet3D(pretrained=False, num_classes=1)
    x = torch.randn(2, 1, 96, 96, 96)
    out = model(x)
    assert out.shape == (2, 1)


@pytest.mark.parametrize("backbone_name", ["CNN", "DenseNet", "ResNet", "EfficientNet"])
def test_feature_extractor_factory_builds_each_backbone(backbone_name: str) -> None:
    factory = FeatureExtractorFactory(feature_dim=512)
    model = factory.build(backbone_name)
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 512)


def test_feature_extractor_factory_rejects_unknown_backbone() -> None:
    factory = FeatureExtractorFactory(feature_dim=512)
    with pytest.raises(ConfigurationError):
        factory.build("NotARealBackbone")


@pytest.mark.parametrize("architecture", ["lstm", "bilstm", "gru", "transformer"])
def test_plaque_sequence_model_output_shape(architecture: str) -> None:
    model = PlaqueSequenceModel(
        input_dim=512, hidden_dim=128, num_layers=1, num_classes=4, architecture=architecture
    )
    x = torch.randn(3, 10, 512)  # batch=3, seq_len=10, feature_dim=512
    out = model(x)
    assert out.shape == (3, 4)


def test_plaque_sequence_model_rejects_unknown_architecture() -> None:
    with pytest.raises(ConfigurationError):
        PlaqueSequenceModel(input_dim=512, hidden_dim=128, num_layers=1, num_classes=4, architecture="rnn-of-doom")
