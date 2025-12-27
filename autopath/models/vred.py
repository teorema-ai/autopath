from dataclasses import dataclass
import functools
import gc
import math
import os
import traceback
import re
from typing import Callable, List, Optional, Tuple

import fsspec

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

import lightning as L
import lightning.pytorch.loggers


import dbx
from dbx import Datablock

from .layers import UpLayer

VERSION = 4

def vector_to_image(vector,):
    N = vector.shape[-1]
    n = int(math.sqrt(N))
    m = int(math.ceil(N/n))
    padsize = (m*n) - N
    if padsize > 0:
        vector = F.pad(vector, (0, padsize))
    image = vector.reshape(vector, n, n)
    return image

class Classifier(nn.Module):
    def __init__(self, 
                 *, 
                 input_dim: int = 1536, 
                 hidden_dim: int = 512, 
                 n_hidden_layers: int = 1, 
                 hidden_activation_cls: Callable = nn.ReLU, 
                 n_classes=100,
                 log: dbx.Logger = None,
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
            self.hidden_activations.append(hidden_activation_cls())
        self.last_layer = nn.Linear(hidden_dim, n_classes)
        self.log = log or dbx.Logger(self.__class__.__name__)

    def forward(self, x):
        self.log.detailed(f"Generating classes for x of type: {type(x)}")
        for i in range(self.n_hidden_layers):
            x = self.hidden_layers[i](x)
            x = self.hidden_activations[i](x)
        x = self.last_layer(x)
        self.log.detailed(f"Generated logits of shape: {x.shape=}")
        x = F.softmax(x, dim=1)
        self.log.detailed(f"Generated classes of shape: {x.shape=}")
        return x
    

class ClassMultiscaleLatentGaussians2D(nn.Module):
    def __init__(self, 
                 *, 
                 n_classes: int = 100, 
                 n_channels: int = 16,
                 input_dim: int = 1536, 
                 hidden_dim: int = 512, 
                 n_hidden_layers: int = 1, 
                 hidden_activation_cls: Callable = nn.ReLU, 
                 fine_scale: int = 256, 
                 n_scales: int = 5,
                 var_min: float = 0.01,
                 var_max: float = None,
                 log: dbx.Logger = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.fine_scale = fine_scale
        self.n_scales = n_scales
        self.var_min = var_min
        self.var_max = var_max
        self.log = log or dbx.Logger(self.__class__.__name__)

        self.latents = nn.ModuleList()
        self.means = nn.ModuleList()
        self.prevariances = nn.ModuleList()
        self.activations = nn.ModuleList()

        self.scales = [fine_scale//(2**i) for i in range(self.n_scales)]
        self.scale_dims = [
            self.n_channels*scale**2 for scale in self.scales
        ]
        self.log.detailed(f"scales: {self.scales}")
        self.log.detailed(f"scale_dims: {self.scale_dims}")
        for scale_dim in self.scale_dims:
            hidden_modules = []
            for i in range(n_hidden_layers):
                N = input_dim + 1 if i == 0 else hidden_dim
                hidden_layer = nn.Linear(N, hidden_dim)
                hidden_activation = hidden_activation_cls()
                hidden_modules.append(hidden_layer)
                hidden_modules.append(hidden_activation)
            self.latents.append(nn.Sequential(*hidden_modules))
            self.means.append(nn.Linear(hidden_dim, scale_dim))
            self.prevariances.append(nn.Linear(hidden_dim, scale_dim))
            if var_max is None:
                self.activations.append(nn.Softplus())
            else:
                self.activations.append(nn.Sigmoid())
        self.log.detailed(f"named_parameters: {list(self.named_parameters())}")

    def forward(self, x, c):
        #c shape (x.shape[0])
        k = c[..., None]
        self.log.detailed(f"forward: ----------------------> {x.shape=}, {c.shape=}, {k.shape=}")
        self.log.detailed(f"forward: devices: -------------> {x.device=}, {c.device=}, {k.device=}")
        u = torch.cat([x, k], dim=-1) # a batch of [vector, scalar_class_idx]
        means_and_variances = []
        for i in range(len(self.latents)):
            latents_devices = {k: v.device for k, v in self.latents[i].named_parameters()}
            self.log.detailed(f"latents: devices: latents[{i}]: devices={latents_devices}, {u.device=}")
            w = self.latents[i](u)
            self.log.detailed(f"means: devices: means[{i}]: {self.means[i].weight.device=}, {w.device=}")
            m = self.means[i](w)
            self.log.detailed(f"prevariances: devices: prevariances[{i}]: {self.prevariances[i].weight.device=}, {w.device=}")
            pv = self.prevariances[i](w)
            amplitude = self.activations[i](pv)
            if self.var_max is None:
                v = amplitude + self.var_min
            else:
                v = (self.var_min + amplitude*self.var_max)
            scale = self.scales[i]
            m = m.reshape(m.shape[0], self.n_channels, scale, scale)
            v = v.reshape(v.shape[0], self.n_channels, scale, scale)
            means_and_variances.append((m, v))
            self.log.detailed(f"Generated latents for scale {self.scales[i]}, {m.shape=}, {v.shape=}")
        return tuple(means_and_variances)
    

class ConvDecoder2D(nn.Module):
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
        multiscale_resolutions: Optional[List[Tuple[int, int]]] = None,
        var_min: float = 0.001,
        var_max: float = None,
        fine_scale_tile_size: int = 256,
        add_skip_features: bool = False,
        log: dbx.Logger = None,
    ):
        super().__init__()

        assert len(skip_features_per_layer) == len(output_features_per_layer)
        self.num_layers = len(skip_features_per_layer)
        in_channels = num_input_features
        self.up_layers = nn.ModuleList()
        for i, (skip_channels, out_channels) in enumerate(
            zip(skip_features_per_layer, output_features_per_layer)
        ):
            layer = UpLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                skip_channels=skip_channels,
                kernel_size=kernel_size,
                use_batch_norm=use_batch_norm,
                add_skip=add_skip_features,
            )
            self.up_layers.append(layer)
            in_channels = out_channels
        self.final_conv = nn.Conv2d(in_channels=output_features_per_layer[-1], out_channels=3, kernel_size=1)
        self.variance_final_conv = nn.Conv2d(in_channels=output_features_per_layer[-1], out_channels=3, kernel_size=1)
        self.variance_final_fc = nn.Linear(fine_scale_tile_size**2*3, 3)
        self.variance_final_activation = nn.Sigmoid() if var_max is not None else nn.Softplus()
        self.multiscale_resolutions = multiscale_resolutions or []
        self.var_min = var_min
        self.var_max = var_max
        self.log = log or dbx.Logger(self.__class__.__name__)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        height = min(f.shape[-2] for f in features)
        mean = torch.cat([f for f in features if f.shape[-2] == height], dim=1)
        self.log.detailed(f"bottom: {height=}, {mean.shape=}")
        self.log.detailed(f"num_layers: {self.num_layers}, top height: {height*(2**self.num_layers)}")

        self.log.detailed(f"feature_shapes: {[f.shape for f in features]}")
        b, c, _, width = mean.shape

        multiscale_features = []
        found_resolutions = []
        for i, up_layer in enumerate(self.up_layers):
            height *= 2
            width *= 2
            skip_features = [torch.empty(b, 0, height, width, device=mean.device)]
            skip_features += [f for f in features if f.shape[-2] == height]
            skip_features = torch.cat(skip_features, dim=1)
            self.log.detailed(f"{height=}, {width=}, {skip_features.shape=}, {mean.shape=}")
            mean = up_layer(mean, skip_features)
            self.log.detailed(f"up_layer_{i}: {mean.shape=}")
            if (height, width) in self.multiscale_resolutions:
                multiscale_features.append(mean)
                found_resolutions.append((height, width))
        assert len(found_resolutions) == len(
            self.multiscale_resolutions
        ), f"Expected multiscale resolutions {self.multiscale_resolutions} but only found {found_resolutions}"
        #

        prevariance = self.variance_final_conv(mean)
        variance_ones = torch.ones(b, 3, height, width).to(mean.device)
        variance_amplitude = self.variance_final_activation(self.variance_final_fc(prevariance.reshape(b, -1))).reshape(b, 3, 1, 1)
        if self.var_max is None:
            variance = (self.var_min + variance_amplitude)*variance_ones
        else:
            variance = (self.var_min + variance_amplitude*self.var_max)*variance_ones

        mean = self.final_conv(mean)
        self.log.detailed(f"final_conv: mean: {mean.shape=}, {mean.device=}")
        self.log.detailed(f"final_conv: variance: {variance.shape=}, {variance.device=}")
        if self.multiscale_resolutions:
            return mean, variance, multiscale_features
        else:
            return mean, variance     


class Loss(nn.Module):
    def forward(self, mean, variance, target):
        loss = torch.sum((mean - target)**2/variance)
        return loss
    

class VariationalReEncoderDecoder(Datablock):
    @dataclass
    class CONFIG:
        classifier_input_dim: int = 1536
        classifier_hidden_dim: int = 512
        classifier_n_hidden_layers: int = 1
        classifier_hidden_activation_cls: Callable = nn.ReLU
        classifier_n_classes: int = 100

        latent_gaussian_n_classes: int = 100
        latent_gaussian_n_channels: int = 16
        latent_gaussian_input_dim: int = 1536
        latent_gaussian_hidden_dim: int = 512
        latent_gaussian_n_hidden_layers: int = 1
        latent_gaussian_hidden_activation_cls: Callable = nn.ReLU
        latent_gaussian_fine_scale: int = 256
        latent_gaussian_n_scales: int = 5
        latent_gaussian_var_min: float = 0.01
        latent_gaussian_var_max: float = None

        decoder_kernel_size: int = 3
        decoder_use_batch_norm: bool = True
        decoder_var_min: float = 0.001
        decoder_var_max: float = 0.1
        loss_var_weight: float = 100.0
        class_batch_size: int = None
        log_mixture_distributions: bool = False
        log_latent_mixture_distributions: bool = False

    class Module(nn.Module):
        INIT_WEIGHTS_STD = 100.0
        def __init__(self, 
                     *, 
                     classifier: Classifier, 
                     latent_gaussians: ClassMultiscaleLatentGaussians2D,
                     decoder: ConvDecoder2D,
                     loss_var_weight: float = 100.0,
                     class_batch_size: int = None,
                     log_mixture_distributions: bool = False,
                     log_latent_mixture_distributions: bool = False,
                     log: dbx.Logger = None,
        ):
            super().__init__()
            self.classifier = classifier
            self.latent_gaussians = latent_gaussians
            self.decoder = decoder
            self.loss_var_weight = loss_var_weight
            self.class_batch_size = class_batch_size
            self.log_mixture_distributions = log_mixture_distributions
            self.log_latent_mixture_distributions = log_latent_mixture_distributions
            self.log = log or dbx.Logger(self.__class__.__name__)
            self.device = 'cpu'
            self.means = None
            self.variances = None
            self.latent_means = None
            self.latent_variances = None

        @staticmethod
        def init_weights(m):
            if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=VariationalReEncoderDecoder.Module.INIT_WEIGHTS_STD)
            if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                torch.nn.init.normal_(m.bias.data, mean=0.0, std=VariationalReEncoderDecoder.Module.INIT_WEIGHTS_STD)

        def to(self, device):
            super().to(device)
            self.device = device
            return self

        @property
        def n_classes(self):
            return self.classifier.n_classes
        
        def sample(self, x):
            self.log.detailed(f"Computing class probabilities for x of shape: {x.shape}")
            class_probabilities = self.classifier(x)
            class_distribution = torch.distributions.Categorical(probs=class_probabilities)
            ksample = class_distribution.sample().to(x.device)
            self.log.detailed(f"Sampled classes {ksample}")
            gaussian_means_and_variances = self.latent_gaussians(x, ksample)
            multiscale_means = []
            for mean, variance in gaussian_means_and_variances:
                normal_sample = torch.randn(mean.shape).to(x.device)
                multiscale_means.append(mean + normal_sample*torch.sqrt(variance))
            decoded_mean, decoded_variance = self.decoder(multiscale_means)
            normal_sample = torch.randn(decoded_mean.shape).to(x.device)
            self.log.detailed(f"sample: {decoded_mean.device=}, {decoded_variance.device=}, {normal_sample.device=}, {x.device=}")
            decoded_sample = decoded_mean + normal_sample*torch.sqrt(decoded_variance)
            return decoded_sample

        def loss(self, x, y):
            self.log.detailed(f"Computing loss for x,y of shapes: {x.shape=}, {y.shape=}, devices: {x.device=}, {y.device=}")
            classes = torch.tensor(list(range(self.n_classes))).to(x.device)
            class_probabilities = self.classifier(x).reshape(1, -1)
            _losses = []
            if self.log_mixture_distributions:
                means_list = []
                variances_list = []
            else:
                means_list = None
                variances_list = None
            if self.log_latent_mixture_distributions:
                latent_means_list = []
                latent_variances_list = []
            else:
                latent_means_list = None
                latent_variances_list = None
            if self.class_batch_size is None:
                self.class_batch_size = self.n_classes
            for class_lo in range(0, self.n_classes, self.class_batch_size):
                class_hi = min(self.n_classes, class_lo + self.class_batch_size)
                class_probabilities_batch = class_probabilities[:, class_lo:class_hi]
                classes_batch = classes[class_lo:class_hi]
                _loss = self._class_batch_loss(x, y, 
                                               classes_batch, class_probabilities_batch, 
                                               means_list=means_list, variances_list=variances_list,
                                               latent_means_list=latent_means_list, latent_variances_list=latent_variances_list
                )
                self.log.detailed(f"loss: _loss: -------------requires_grad ------------> {_loss.requires_grad}")
                _losses.append(_loss)
            losses = torch.stack(_losses)
            self.log.detailed(f"loss: losses: -------------requires_grad ------------> {losses.requires_grad}")
            loss = torch.sum(losses, dim=0) #TODO: take .mean()?
            if self.log_mixture_distributions:
                self.means = torch.cat(means_list, dim=0)
                self.variances = torch.cat(variances_list, dim=0)
                del means_list
                del variances_list
            if self.log_latent_mixture_distributions:
                self.latent_means = torch.cat(latent_means_list, dim=0)
                self.latent_variances = torch.cat(latent_variances_list, dim=0)
                del latent_means_list
                del latent_variances_list
            del losses
            del class_probabilities
            del classes
            gc.collect()
            torch.cuda.empty_cache()
            self.log.detailed(f"loss: loss: -------------requires_grad ------------> {loss.requires_grad}")
            return loss

        def _class_batch_loss(self, x, y, classes, class_probabilities, *, means_list=None, variances_list=None, latent_means_list=None, latent_variances_list=None):
            b = x.shape[0]
            k = classes.shape[0]
            C = classes.reshape(-1, 1).repeat(1, b).reshape(b*k)
            X = x.repeat(self.n_classes, *([1]*len(x.shape[1:])))
            gaussian_means_and_variances = self.latent_gaussians(X, C)
            del X
            del C
            gc.collect()
            torch.cuda.empty_cache()
            multiscale_means = []
            for scale, (mean, variance) in enumerate(gaussian_means_and_variances):
                normal_sample = torch.randn(mean.shape).to(x.device)
                multiscale_means.append(mean + normal_sample*torch.sqrt(variance))
            if self.log_latent_mixture_distributions:
                latent_means, latent_variances = zip(*gaussian_means_and_variances)
                latent_mean = torch.cat(latent_means, dim=0)
                latent_variance = torch.cat(latent_variances, dim=0)
                latent_means_list.append(latent_mean)
                latent_variances_list.append(latent_variance)
            Mhat, Vhat = self.decoder(multiscale_means)
            self.log.detailed(f"_class_batch_loss: computing loss for {len(classes)} classes, obtained {len(Mhat)} Mhat from {len(multiscale_means)} multiscale means")
            if self.log_mixture_distributions:
                means_list.append(Mhat)
                variances_list.append(Vhat)
            del multiscale_means
            gc.collect()
            torch.cuda.empty_cache()
            """
            # mhat: "(k b) c h w" --> yhat: "k b c h w"
            x = [
                [1., 2.],
                [3., 4.]
            ]
            #
            k = [0, 1, 3]
            #
            XK = [[[1., 2., 0.],
                   [3., 4., 0.],
                   [1., 2., 1.],
                   [3., 4., 1.],
                   [1., 2., 2.],
                   [3., 4., 2.]]]
            """
            Y = y.repeat(self.n_classes, *([1]*len(y.shape[1:]))).to(x.dtype)

            diffsquared = (Mhat - Y)**2
            diffscaled = diffsquared/Vhat
            _loss_ = torch.sqrt(diffscaled) + 0.5*torch.log(Vhat)*self.loss_var_weight # (k b) c h w
            _loss = torch.sum(_loss_, dim=(1, 2, 3)) # (k b)
            _loss_nans = torch.isnan(_loss).sum().item()
            _loss_nans_ = torch.isnan(_loss_).sum().item()
            self.log.detailed(f"_class_batch_loss: _loss_nans: {_loss_nans}, _loss_nans_: {_loss_nans_}")
            del Y
            del _loss_
            gc.collect()
            torch.cuda.empty_cache()
            _kloss = _loss.reshape(k, b) # k b
            _loss = torch.matmul(class_probabilities, _kloss) # b
            loss = _loss.mean() # scalar
            del _loss
            del _kloss
            gc.collect()
            torch.cuda.empty_cache()
            return loss

    def model(self) -> nn.Module:
        classifier = Classifier(
            input_dim=self.cfg.classifier_input_dim,
            hidden_dim=self.cfg.classifier_hidden_dim,
            n_hidden_layers=self.cfg.classifier_n_hidden_layers,
            hidden_activation_cls=self.cfg.classifier_hidden_activation_cls,
            n_classes=self.cfg.classifier_n_classes,
            log=self.log,
        )
        latent_gaussians = ClassMultiscaleLatentGaussians2D(
            n_classes=self.cfg.latent_gaussian_n_classes,
            n_channels=self.cfg.latent_gaussian_n_channels,
            input_dim=self.cfg.latent_gaussian_input_dim,
            hidden_dim=self.cfg.latent_gaussian_hidden_dim,
            n_hidden_layers=self.cfg.latent_gaussian_n_hidden_layers,
            hidden_activation_cls=self.cfg.latent_gaussian_hidden_activation_cls,
            fine_scale=self.cfg.latent_gaussian_fine_scale,
            n_scales=self.cfg.latent_gaussian_n_scales,
            var_min=self.cfg.latent_gaussian_var_min,
            var_max=self.cfg.latent_gaussian_var_max,
            log=self.log,
        )
        assert classifier.n_classes == latent_gaussians.n_classes, f"Classifier has {classifier.n_classes} classes but latent_gaussians has {latent_gaussians.n_classes} classes"
        decoder = ConvDecoder2D(
            num_input_features=latent_gaussians.n_channels,
            skip_features_per_layer=[latent_gaussians.n_channels]*(latent_gaussians.n_scales-2) + [latent_gaussians.n_channels],
            output_features_per_layer=[latent_gaussians.n_channels]*(latent_gaussians.n_scales-2) + [latent_gaussians.n_channels],
            kernel_size=self.cfg.decoder_kernel_size,
            use_batch_norm=self.cfg.decoder_use_batch_norm,
            var_min=self.cfg.decoder_var_min,
            var_max=self.cfg.decoder_var_max,
            fine_scale_tile_size=latent_gaussians.fine_scale,
            log=self.log,
        )
        return self.Module(
            classifier=classifier,
            latent_gaussians=latent_gaussians,
            decoder=decoder,
            loss_var_weight=self.cfg.loss_var_weight,
            class_batch_size=self.cfg.class_batch_size,
            log_mixture_distributions=self.cfg.log_mixture_distributions,
            log_latent_mixture_distributions=self.cfg.log_latent_mixture_distributions,
            log=self.log,
        )


class VariationalReEncoderDecoderEvaluator(Datablock):
    @dataclass
    class CONFIG:
        vred: VariationalReEncoderDecoder
        dataloader: torch.utils.data.DataLoader

    def __post_init__(self):
        self.vred = self.cfg.vred.model()
        self.dataloader = self.cfg.dataloader

    def to(self, device):
        self.device = device
        self.vred.to(device)
        return self

    def samples(self, n_batches: int = 1):
        self.log.debug(f"Sampling {n_batches} batches")
        output_batches_list = []
        batchiter = iter(self.dataloader)
        for _ in range(n_batches):
            batch = next(batchiter)
            _input_features = batch[0] 
            _output_batch = self.vred.sample(_input_features.to(self.device)).to('cpu')
            output_batches_list.append(_output_batch)
        return output_batches_list
    
    def losses(self, n_batches: int = 1):
        self.log.debug(f"Computing losses for {n_batches} batches")
        loss_list = []
        batchiter = iter(self.dataloader)
        for _ in range(n_batches):
            batch = next(batchiter)
            _input_features = batch[0] 
            _input_tiles = batch[1][1]
            _loss = self.vred.loss(_input_features.to(self.device), _input_tiles.to(self.device)).to('cpu')
            loss_list.append(_loss)
        return loss_list


class VariationalReEncoderDecoderLightning(Datablock):
    VERSION = globals().get('VERSION', None)
    @dataclass
    class CONFIG:
        vred: VariationalReEncoderDecoder
        learning_rate: float = 1e-3
        scheduler: str = "cosine"
        log_tiles: bool = False
        log_feature_norms: bool = False

    class Callbacks(L.pytorch.callbacks.Callback):
        def __init__(self, 
                     *, 
                     skip_invalid_gradients: bool = True,
                     log_gradients: bool = False, 
                     log_weights: bool = False, 
                     log: dbx.Logger = dbx.Logger(name="VariationalReEncoderDecoderLightning.Callbacks")
        ):
            super().__init__()
            self.log_gradients = log_gradients
            self.log_weights = log_weights
            self.skip_invalid_gradients = skip_invalid_gradients
            self.log = log

        def on_after_backward(self, trainer, module):
            step = module.global_step
            if self.log_gradients or self.log_weights:
                for name, param in module.vred.named_parameters():
                    if param.grad is not None:
                        try:
                            module.logger.experiment.add_histogram(f"grad/{name}/{step=}", param.grad, module.global_step)
                        except Exception as e:
                            tbstr = '\n'.join(traceback.format_tb(e.__traceback__))
                            self.log.info(f"on_after_backward: param: grad: {name}: {e}\n{tbstr}")
                    try:
                        self.log.detailed(f"Logging weights for {name} ... ")
                        module.logger.experiment.add_histogram(f"{name}/{step=}", param, module.global_step)
                        self.log.detailed(f"Logging weights for {name} ... done")
                    except Exception as e:
                        tbstr = '\n'.join(traceback.format_tb(e.__traceback__))
                        self.log.info(f"on_after_backward: param: {name}: {e}\n{tbstr}")
            if self.skip_invalid_gradients:
                gradients_valid = True
                for name, param in module.vred.named_parameters():    
                    if param.grad is not None:
                        self.log.detailed(f"Checking for invalid gradients in {name} ... ")
                        self.log.detailed(f"Checking for invalid gradients in {name} ... done")
                        gradients_valid = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                        if not gradients_valid:
                            break
                if not gradients_valid:
                    self.log.info(f"on_after_backward: skipping invalid gradients for step {step}")
                    module.zero_grad()                    

    class Lightning(L.LightningModule):
        def __init__(self, vred: VariationalReEncoderDecoder.Module, learning_rate: float = 1e-3, scheduler: str = "cosine", log_tiles: bool = False, log_feature_norms: bool = False, log: dbx.Logger = dbx.Logger(name="Lightning")):
            super().__init__()
            self.vred = vred
            self.learning_rate = learning_rate
            self.scheduler = scheduler
            self.save_hyperparameters(ignore=['vred'])
            self.log_tiles = log_tiles
            self.log_feature_norms = log_feature_norms
            self.log = log
                                         
        def training_step(self, batch, batch_idx):
            features, labels = batch
            bag, tiles = labels
            loss = self.vred.loss(features, tiles)
            self.logger.experiment.add_scalar(f"Loss", loss, self.global_step)
            scheduler = self.lr_schedulers()
            lr = scheduler.get_last_lr()[0]
            self.logger.experiment.add_scalar(f"Learning Rate", lr, self.global_step)
            b = features.shape[0]
            i = np.random.randint(b)
            if self.log_tiles:
                self.logger.experiment.add_image(f"Tile", tiles[i], self.global_step)
            if self.vred.log_mixture_distributions:
                for k in range(self.vred.n_classes):
                    mean = self.vred.means[i*self.vred.n_classes+k].squeeze()
                    variance_matrix = self.vred.variances[i*self.vred.n_classes+k].squeeze()
                    variance_vector = variance_matrix.mean(dim=(-1, -2))
                    self.log.detailed(f"Logging mixture distribution for class {k}: {mean.shape=}, {variance_matrix.shape=}, {variance_vector=}")
                    feature_norms = [torch.linalg.norm(features[i]) for i in range(len(features))]
                    self.logger.experiment.add_image(f"mix_mean/component={k}", mean, self.global_step)
                    for s in range(3):
                        self.logger.experiment.add_scalar(f"mix_variance/component={k}/{s}", variance_vector[s], self.global_step)
            if self.vred.log_latent_mixture_distributions:
                for k in range(self.vred.n_classes):
                    for s in range(self.vred.latent_gaussians.n_scales):
                        latent_mean = self.vred.latent_means[i*self.vred.n_classes+k, s].squeeze()
                        latent_variance = self.vred.latent_variances[i*self.vred.n_classes+k, s].squeeze()
                        self.log.detailed(f"Logging latent mixture distribution for class {k}, scale {s}: {latent_mean.shape=}, {latent_variance.shape=}")
                        latent_mean_image = vector_to_image(latent_mean)
                        latent_variance_image = vector_to_image(latent_variance)
                        self.logger.experiment.add_image(f"latent_mix_mean/component={k}", latent_mean_image, self.global_step)
                        self.logger.experiment.add_scalar(f"latent_mix_variance/component={k}/{s}", latent_variance_image, self.global_step)
            if self.log_feature_norms:
                feature_norms = [torch.linalg.norm(features[i]) for i in range(len(features))]
                self.logger.experiment.add_scalar(f"feature_norms_max", max(feature_norms), self.global_step)
                self.logger.experiment.add_scalar(f"feature_norms_min", min(feature_norms), self.global_step)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.vred.parameters(), lr=self.learning_rate)
            stepping_batches = self.trainer.estimated_stepping_batches
            if self.scheduler == "onecyclelr":
                
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=stepping_batches)
            elif self.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    #max_lr=self.learning_rate,
                    T_max=stepping_batches,
                    eta_min=1e-6
                )
            else: 
                raise ValueError(f"Unknown scheduler: {self.scheduler}")
            self.log.verbose(f"Using learning rate scheduler: {self.scheduler}")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    @functools.cached_property
    def lightning_module(self):
        return self.Lightning(vred=self.cfg.vred.model(), learning_rate=self.cfg.learning_rate, scheduler=self.cfg.scheduler, log_tiles=self.cfg.log_tiles, log_feature_norms=self.cfg.log_feature_norms)
    

class VariationalReEncoderDecoderStill(Datablock):
    VERSION = globals().get('VERSION', None)
    TOPICFILES = {'logs': None,
                  'ckpts': None,
    }
    
    @dataclass 
    class CONFIG:
        lightning: VariationalReEncoderDecoderLightning
        dataloader: torch.utils.data.DataLoader
        init_ckpt_path_or_anchor: str = None
        restart: bool = False
        load_optimizer_state: bool = False
        max_epochs: int = 1
        max_steps: int = 1
        log_interval: int = 1
        log_gradients: bool = False
        log_weights: bool = False
        skip_invalid_gradients: bool = True
        gradient_clip_val: float = 1.0
        gradient_clip_algorithm: str = "norm"
        ckpt_every_n_steps: int = None
        precision: str = None

    def __init__(self, *args, n_devices: int = 1, logs: str = None, **kwargs):
        super().__init__(*args, n_devices=n_devices, logs=logs, **kwargs)

    def __pre_build__(self):
        super().__pre_build__()
        self.linklogs()
        return self

    def linklogs(self):
        # Link the logs directory to the provided location (e.g., for Tensorboard to pick up the logs)
        if self.logs is not None:
            self.log.verbose(f"---------------------- Linking logs to {self.logs}----------------------------")
            try: #os.path.exists(self.logs) can sometimes return False when the link actually exists
                self.log.debug(f"Removing {self.logs} link")
                os.remove(self.logs)
            except:
                pass
            os.symlink(self.dirpath('logs', ensure=True), self.logs)
        return self

    def valid(self):
        #TODO: check if max_epochs has been run and a corresponding ckpt has been generated
        return False
    
    def ckpt(self):
        dirpath = self.dirpath('ckpts')
        ckptfs, _ = fsspec.url_to_fs(dirpath)
        files = [] if not ckptfs.exists(dirpath) else ckptfs.ls(dirpath)
        ckpts = [f for f in files if f.endswith('.ckpt') and 'step=' in f]
        steps = []
        for ckpt in ckpts:
            _, basename = os.path.split(ckpt)
            name, _ = basename.split('.')
            _, stepstr = name.split('step=')
            step = int(stepstr)
            steps.append(step)
        if len(steps) == 0:
            ckpt = None
        else:
            i = np.argmax(np.array(steps))
            ckpt = ckpts[i]
        return ckpt 
    
    def __build__(self):
        logger = L.pytorch.loggers.TensorBoardLogger(save_dir=self.dirpath('logs'), default_hp_metric=False, name=self.anchor)
        default_root_dir = self.dirpath('ckpts')
        self.log.detailed(f"Built {logger=} for lightining {self.cfg.lightning}")
        self.log.debug(f"Building trainer using {default_root_dir=} to train for {self.cfg.max_steps=} using {self.cfg.lightning=} and {self.n_devices=}")
        kwargs = {}
        if self.cfg.gradient_clip_val > 0.0:
            kwargs['gradient_clip_val'] = self.cfg.gradient_clip_val
            self.log.info(f"----------> Using gradient clipping with value {self.cfg.gradient_clip_val} and algorithm {self.cfg.gradient_clip_algorithm} <----------")
        callbacks = [
                VariationalReEncoderDecoderLightning.Callbacks(log_gradients=self.cfg.log_gradients, 
                                                               log_weights=self.cfg.log_weights, 
                                                               skip_invalid_gradients=self.cfg.skip_invalid_gradients),
                
            ]
        if self.cfg.ckpt_every_n_steps is not None:
            callbacks.append(L.pytorch.callbacks.ModelCheckpoint(dirpath=self.dirpath('ckpts'), every_n_train_steps=self.cfg.ckpt_every_n_steps))
        trainer = L.pytorch.Trainer(
            default_root_dir=default_root_dir, 
            max_epochs=self.cfg.max_epochs, 
            limit_train_batches=self.cfg.max_steps,
            log_every_n_steps=self.cfg.log_interval,
            callbacks=callbacks,
            devices=self.n_devices,
            logger=logger,
            **kwargs,
        )
        self.log.debug(f"Built {trainer=} for lightining {self.cfg.lightning}")
        self.log.debug(f"Launching training for {self.cfg.max_steps=}")
        original_precision = torch.get_float32_matmul_precision()
        if self.cfg.precision is not None:
            self.log.info(f"Setting precision to {repr(self.cfg.precision)}")
            torch.set_float32_matmul_precision(self.cfg.precision)
        try:
            model = self.cfg.lightning.lightning_module
            resume = not self.cfg.restart
            ckpt = None
            if resume: 
                ckpt = self.ckpt()
            if ckpt is None:
                if self.cfg.init_ckpt_path_or_anchor is not None:
                    ckpath = self.cfg.init_ckpt_path_or_anchor
                    if ckpath.startswith('/'):  #TODO: support for fsspec urls
                        ckpt = ckpath
                    else:
                        ckpt = os.path.join(self.root, ckpath)
            fit_kwargs = {}
            if ckpt is not None:
                self.log.info(f"Using checkpoint {ckpt}")
                if not self.cfg.load_optimizer_state:
                    self.log.info(f"Skipping optimizer state from {ckpt}")
                    checkpoint = torch.load(ckpt, weights_only=False)
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    fit_kwargs['ckpt_path'] = ckpt
            trainer.fit(model=model, train_dataloaders=self.cfg.dataloader, **fit_kwargs)
        finally:
            torch.set_float32_matmul_precision(original_precision)
        return self
    