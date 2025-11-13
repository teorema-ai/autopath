import copy
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import UpLayer


class Classifier(nn.Module):
    def __init__(self, 
                 *, 
                 input_dim: int = 1536, 
                 hidden_dim: int = 512, 
                 n_hidden_layers: int = 1, 
                 n_hidden_activation_cls: Callable = nn.ReLU, 
                 n_classes=100
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.n_hidden_layers = n_hidden_layers

        self.hidden_layers = nn.ModuleList()
        self.hidden_activations = nn.ModuleList()
        for i in range(n_hidden_layers):
            N = input_dim if i == 0 else hidden_dim
            self.hidden_layers.append(nn.Linear(N, hidden_dim))
            self.hidden_activations.append(n_hidden_activation_cls())
        self.last_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = self.hidden_layers[i](x)
            x = self.hidden_activations[i](x)
        x = self.last_layer(x)
        x = F.softmax(x, dim=1)
        return x
    

class ClassLatentGaussians(nn.Module):
    def __init__(self, 
                 *, 
                 n_classes: int = 100, 
                 hidden_dim: int = 512, 
                 n_hidden_layers: int = 1, 
                 n_hidden_activation_cls: Callable = nn.ReLU, 
                 latent_dim: int = 1536,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        self.hidden_layers = nn.ModuleList()
        self.hidden_activations = nn.ModuleList()
        for i in range(n_hidden_layers):
            N = n_classes if i == 0 else hidden_dim
            self.hidden_layers.append(nn.Linear(N, hidden_dim))
            self.hidden_activations.append(n_hidden_activation_cls())
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.prevariance = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        for i in range(self.n_hidden_layers):
            x = self.hidden_layers[i](x)
            x = self.hidden_activations[i](x)
        m = self.mean(x)
        v = F.softplus(self.prevariance(x))
        return m, v


class ConvDecoder(nn.Module):
    """Simple convolutional pixel decoder.

    Accepts a list of feature maps with the same aspect ratio and
    combines them into a full resolution feature map using convolutional
    layers. Similar to a UNet decoder, start with the lowest resolution
    feature maps and progressively upscale + concatenate.

    Args:
        num_input_features: Number of input features in the lowest resolution.
            This should be the sum of channels over all feature maps of this
            resolution.
        skip_features_per_layer: List of number of skip features for each resolution
            between the lowest and full resolution. If any resolutions do not have
            feature maps, then this should be set to 0 for that list item. Lowest
            resolution should not be included as it is already taken care of by
            `num_input_features`.
        output_features_per_layer: List of number of output features per layer.
            Should have the same length as `skip_features_per_layer` but otherwise
            no constraints.
        kernel_size: Convolutional kernel size.
        use_batch_norm: Whether to use batch norm.
    """

    def __init__(
        self,
        num_input_features: int,
        skip_features_per_layer: List[int],
        output_features_per_layer: List[int],
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        variance_scale: float = 0.03,
        multiscale_resolutions: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()

        assert len(skip_features_per_layer) == len(output_features_per_layer)
        self.num_layers = len(skip_features_per_layer)
        in_channels = num_input_features
        for i, (skip_channels, out_channels) in enumerate(
            zip(skip_features_per_layer, output_features_per_layer)
        ):
            layer = UpLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                skip_channels=skip_channels,
                kernel_size=kernel_size,
                use_batch_norm=use_batch_norm,
            )
            layer_idx = self.num_layers - i - 1
            setattr(self, f"up_layer_{layer_idx}", layer)
            in_channels = out_channels
        self.multiscale_resolutions = multiscale_resolutions or []

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        height = min(f.shape[-2] for f in features)
        mean = torch.cat([f for f in features if f.shape[-2] == height], dim=1)
        bs, _, _, width = mean.shape

        multiscale_features = []
        found_resolutions = []
        for i in reversed(list(range(self.num_layers))):
            height *= 2
            width *= 2
            skip_features = [torch.empty(bs, 0, height, width, device=mean.device)]
            skip_features += [f for f in features if f.shape[-2] == height]
            skip_features = torch.cat(skip_features, dim=1)
            up_layer = getattr(self, f"up_layer_{i}")
            mean = up_layer(mean, skip_features)
            if (height, width) in self.multiscale_resolutions:
                multiscale_features.append(mean)
                found_resolutions.append((height, width))
        assert len(found_resolutions) == len(
            self.multiscale_resolutions
        ), f"Expected multiscale resolutions {self.multiscale_resolutions} but only found {found_resolutions}"
        variance = torch.full((bs, mean.shape[1], height, width), self.variance_scale)
        if self.multiscale_resolutions:
            return mean, variance, multiscale_features
        else:
            return mean, variance


class Loss(nn.Module):
    def forward(self, mean, variance, target):
        loss = torch.sum((mean - target)**2/variance)
        return loss
    

class VariationalReDecoder:
    ...