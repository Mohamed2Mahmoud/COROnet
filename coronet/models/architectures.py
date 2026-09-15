"""3D CNN model architectures used for vessel classification and feature extraction.

Refactored from ``05A_resNet75.ipynb`` (binary vessel classifier) and
``05B_Feature_Extractor.ipynb`` (multi-backbone 512-d feature
extractors).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torchvision.models.video as video_models
from monai.networks.nets import DenseNet121, EfficientNetBN, resnet18

from coronet.exceptions import ConfigurationError
from coronet.logging_config import get_logger

logger = get_logger(__name__)


class CustomCNN3D(nn.Module):
    """A lightweight custom 3D CNN that compresses a 32x32x32 patch to a 512-d vector.

    Architecture: three ``Conv3d -> ReLU -> MaxPool3d`` blocks followed
    by a linear projection to the target feature dimension.

    Parameters
    ----------
    output_dim:
        Dimensionality of the output feature vector.
    """

    def __init__(self, output_dim: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, 1, 32, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Output feature tensor of shape ``(batch, output_dim)``.
        """
        return self.features(x)


class VesselResNet3D(nn.Module):
    """A 3D ResNet-18 (video architecture) adapted for single-channel binary vessel classification.

    Wraps ``torchvision.models.video.r3d_18``, replacing its stem
    convolution to accept single-channel (grayscale CT) input and its
    final fully connected layer to output a single logit.

    Parameters
    ----------
    pretrained:
        If ``True``, initialize from Kinetics-400 pretrained weights
        before adapting the stem/head. Ignored when ``num_classes`` is
        loaded from a checkpoint immediately afterward.
    num_classes:
        Number of output logits (``1`` for binary classification with a
        sigmoid).
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 1) -> None:
        super().__init__()
        weights = "KINETICS400_V1" if pretrained else None
        self.backbone = video_models.r3d_18(weights=weights)

        old_conv = self.backbone.stem[0]
        self.backbone.stem[0] = nn.Conv3d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, 1, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(batch, num_classes)``.
        """
        return self.backbone(x)


class FeatureExtractorFactory:
    """Builds the set of 3D CNN backbones used for multi-model feature extraction.

    Parameters
    ----------
    feature_dim:
        Output dimensionality shared by every backbone.
    """

    _SUPPORTED_BACKBONES = ("CNN", "DenseNet", "ResNet", "EfficientNet")

    def __init__(self, feature_dim: int = 512) -> None:
        self.feature_dim = feature_dim

    def build(self, backbone_name: str) -> nn.Module:
        """Instantiate a single named feature-extraction backbone.

        Parameters
        ----------
        backbone_name:
            One of ``"CNN"``, ``"DenseNet"``, ``"ResNet"``,
            ``"EfficientNet"``.

        Returns
        -------
        torch.nn.Module
            A model mapping a ``(batch, 1, D, H, W)`` patch batch to a
            ``(batch, feature_dim)`` feature batch.

        Raises
        ------
        ConfigurationError
            If ``backbone_name`` is not supported.
        """
        if backbone_name == "CNN":
            return CustomCNN3D(output_dim=self.feature_dim)
        if backbone_name == "DenseNet":
            return DenseNet121(spatial_dims=3, in_channels=1, out_channels=self.feature_dim)
        if backbone_name == "ResNet":
            return resnet18(spatial_dims=3, n_input_channels=1, num_classes=self.feature_dim)
        if backbone_name == "EfficientNet":
            return EfficientNetBN(
                "efficientnet-b0", spatial_dims=3, in_channels=1, num_classes=self.feature_dim
            )
        raise ConfigurationError(
            f"Unsupported backbone '{backbone_name}'. Supported: {self._SUPPORTED_BACKBONES}"
        )

    def build_all(self) -> Dict[str, nn.Module]:
        """Instantiate every supported backbone.

        Returns
        -------
        dict of str to torch.nn.Module
            Mapping from backbone name to an initialized (untrained,
            evaluation-ready) model instance.
        """
        return {name: self.build(name) for name in self._SUPPORTED_BACKBONES}
