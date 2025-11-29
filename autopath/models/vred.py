from dataclasses import dataclass
import functools
import gc
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
import lightning.pytorch.loggers


import dbx
from dbx import Datablock

from .layers import UpLayer


class Classifier(nn.Module):
    def __init__(self, 
                 *, 
                 input_dim: int = 1536, 
                 hidden_dim: int = 512, 
                 n_hidden_layers: int = 1, 
                 n_hidden_activation_cls: Callable = nn.ReLU, 
                 n_classes=100,
                 log: dbx.Logger = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.n_hidden_layers = n_hidden_layers
        self.log = dbx.Logger(self.__class__.__name__)

        self.hidden_layers = nn.ModuleList()
        self.hidden_activations = nn.ModuleList()
        for i in range(n_hidden_layers):
            N = input_dim if i == 0 else hidden_dim
            self.hidden_layers.append(nn.Linear(N, hidden_dim))
            self.hidden_activations.append(n_hidden_activation_cls())
        self.last_layer = nn.Linear(hidden_dim, n_classes)
        self.log = log or dbx.Logger(self.__class__.__name__)

    def forward(self, x):
        self.log.debug(f"Generating classes for x of type: {type(x)}")
        for i in range(self.n_hidden_layers):
            x = self.hidden_layers[i](x)
            x = self.hidden_activations[i](x)
        x = self.last_layer(x)
        self.log.debug(f"Generated logits of shape: {x.shape=}")
        x = F.softmax(x, dim=1)
        self.log.debug(f"Generated classes of shape: {x.shape=}")
        return x
    

class ClassMultiscaleLatentGaussians2D(nn.Module):
    def __init__(self, 
                 *, 
                 n_classes: int = 100, 
                 n_channels: int = 16,
                 input_dim: int = 1536, 
                 hidden_dim: int = 512, 
                 n_hidden_layers: int = 1, 
                 n_hidden_activation_cls: Callable = nn.ReLU, 
                 fine_scale: int = 256, 
                 n_scales: int = 5,
                 variance_eps: float = 0.01,
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
        self.variance_eps = variance_eps
        self.log = log or dbx.Logger(self.__class__.__name__)

        self.latents = nn.ModuleList()
        self.means = nn.ModuleList()
        self.prevariances = nn.ModuleList()
        self.softpluses = nn.ModuleList()

        self.scales = [fine_scale//(2**i) for i in range(self.n_scales)]
        self.scale_dims = [
            self.n_channels*scale**2 for scale in self.scales
        ]
        self.log.debug(f"scales: {self.scales}")
        self.log.debug(f"scale_dims: {self.scale_dims}")
        for scale_dim in self.scale_dims:
            hidden_modules = []
            for i in range(n_hidden_layers):
                N = input_dim + 1 if i == 0 else hidden_dim
                hidden_layer = nn.Linear(N, hidden_dim)
                hidden_activation = n_hidden_activation_cls()
                hidden_modules.append(hidden_layer)
                hidden_modules.append(hidden_activation)
            self.latents.append(nn.Sequential(*hidden_modules))
            self.means.append(nn.Linear(hidden_dim, scale_dim))
            self.prevariances.append(nn.Linear(hidden_dim, scale_dim))
            self.softpluses.append(nn.Softplus())
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
            v = self.softpluses[i](pv) + self.variance_eps
            scale = self.scales[i]
            m = m.reshape(m.shape[0], self.n_channels, scale, scale)
            v = v.reshape(v.shape[0], self.n_channels, scale, scale)
            means_and_variances.append((m, v))
            self.log.debug(f"Generated latents for scale {self.scales[i]}, {m.shape=}, {v.shape=}")
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
        variance_scale: float = 0.03,
        add_skip_features: bool = False,
        log: dbx.Logger = None,
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
                add_skip=add_skip_features,
            )
            layer_idx = self.num_layers - i - 1
            setattr(self, f"up_layer_{layer_idx}", layer) #TODO: use nn.ModuleList
            in_channels = out_channels
        self.final_conv = nn.Conv2d(in_channels=output_features_per_layer[-1], out_channels=3, kernel_size=1)
        self.multiscale_resolutions = multiscale_resolutions or []
        self.variance_scale = variance_scale
        self.log = log or dbx.Logger(self.__class__.__name__)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        height = min(f.shape[-2] for f in features)
        mean = torch.cat([f for f in features if f.shape[-2] == height], dim=1)
        self.log.debug(f"bottom: {height=}, {mean.shape=}")
        self.log.debug(f"num_layers: {self.num_layers}, top height: {height*(2**self.num_layers)}")

        self.log.debug(f"feature_shapes: {[f.shape for f in features]}")
        bs, _, _, width = mean.shape

        multiscale_features = []
        found_resolutions = []
        for i in reversed(list(range(self.num_layers))):
            height *= 2
            width *= 2
            skip_features = [torch.empty(bs, 0, height, width, device=mean.device)]
            skip_features += [f for f in features if f.shape[-2] == height]
            skip_features = torch.cat(skip_features, dim=1)
            up_layer = getattr(self, f"up_layer_{i}") # TODO: use a nn.ModuleList
            self.log.debug(f"{height=}, {width=}, {skip_features.shape=}, {mean.shape=}")
            mean = up_layer(mean, skip_features)
            self.log.debug(f"up_layer_{i}: {mean.shape=}")
            if (height, width) in self.multiscale_resolutions:
                multiscale_features.append(mean)
                found_resolutions.append((height, width))
        assert len(found_resolutions) == len(
            self.multiscale_resolutions
        ), f"Expected multiscale resolutions {self.multiscale_resolutions} but only found {found_resolutions}"
        mean = self.final_conv(mean)
        self.log.debug(f"final_conv: mean: {mean.shape=}, {mean.device=}")
        variance = torch.full((bs, mean.shape[1], height, width), self.variance_scale).to(mean.device)
        self.log.debug(f"final_conv: variance: {variance.shape=}, {variance.device=}")
        if self.multiscale_resolutions:
            return mean, variance, multiscale_features
        else:
            return mean, variance     


class Loss(nn.Module):
    def forward(self, mean, variance, target):
        loss = torch.sum((mean - target)**2/variance)
        return loss
    

class VariationalReDecoder(nn.Module):
    def __init__(self, 
                 *, 
                 classifier: Classifier, 
                 latent_gaussians: ClassMultiscaleLatentGaussians2D,
                 kernel_size: int = 3, 
                 use_batch_norm: bool = True,
                 variance_scale: float = 0.03,
                 class_batch_size: int = None,
                 log: dbx.Logger = None,
    ):
        super().__init__()
        self.classifier = dbx.eval_term(classifier)
        self.latent_gaussians = dbx.eval_term(latent_gaussians)
        assert self.classifier.n_classes == self.latent_gaussians.n_classes, f"Classifier has {self.classifier.n_classes} classes but latent_gaussians has {self.latent_gaussians.n_classes} classes"
        self.decoder = ConvDecoder2D(
            num_input_features=self.latent_gaussians.n_channels,
            skip_features_per_layer=[self.latent_gaussians.n_channels]*(self.latent_gaussians.n_scales-2) + [self.latent_gaussians.n_channels],
            output_features_per_layer=[self.latent_gaussians.n_channels]*(self.latent_gaussians.n_scales-2) + [self.latent_gaussians.n_channels],
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            variance_scale=variance_scale,
        )
        self.class_batch_size = class_batch_size
        self.log = log or dbx.Logger(self.__class__.__name__)
        self.device = 'cpu'

    def to(self, device):
        self.classifier.to(device)
        self.latent_gaussians.to(device)
        self.decoder.to(device)
        self.device = device
        return self

    @property
    def n_classes(self):
        return self.classifier.n_classes
    
    def sample(self, x):
        self.log.debug(f"Computing class probabilities for x of shape: {x.shape}")
        class_probabilities = self.classifier(x)
        class_distribution = torch.distributions.Categorical(probs=class_probabilities)
        ksample = class_distribution.sample().to(x.device)
        self.log.debug(f"Sampled classes {ksample}")
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
        self.log.debug(f"Computing loss for x,y of shapes: {x.shape=}, {y.shape=}, devices: {x.device=}, {y.device=}")
        classes = torch.tensor(list(range(self.n_classes))).to(x.device)
        class_probabilities = self.classifier(x).reshape(1, -1)
        losses = []
        if self.class_batch_size is None:
            self.class_batch_size = self.n_classes
        for class_lo in range(0, self.n_classes, self.class_batch_size):
            class_hi = min(self.n_classes, class_lo + self.class_batch_size)
            class_probabilities_batch = class_probabilities[:, class_lo:class_hi]
            classes_batch = classes[class_lo:class_hi]
            _loss = self._class_batch_loss(x, y, classes_batch, class_probabilities_batch)
            losses.append(_loss)
        loss = torch.sum(torch.tensor(losses)) #TODO: take .mean()?
        del losses
        del class_probabilities
        del classes
        gc.collect()
        torch.cuda.empty_cache()
        return loss

    def _class_batch_loss(self, x, y, classes, class_probabilities):
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
        for mean, variance in gaussian_means_and_variances:
            normal_sample = torch.randn(mean.shape).to(x.device)
            multiscale_means.append(mean + normal_sample*torch.sqrt(variance))
        Mhat, Vhat = self.decoder(multiscale_means)
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
        _loss_ = torch.sqrt((Mhat - Y)**2/Vhat) # (k b) c h w
        _loss = torch.sum(_loss_, dim=(1, 2, 3)) # (k b)
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
    

class VariationalReDecoderEvaluator(Datablock):
    @dataclass
    class CONFIG:
        vred: VariationalReDecoder
        dataloader: torch.utils.data.DataLoader

    def __post_init__(self):
        self.vred = self.cfg.vred
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


class VariationalReDecoderLightning(Datablock):
    class Lightning(L.LightningModule):
        def __init__(self, vred: VariationalReDecoder, learning_rate: float = 1e-3):
            super().__init__()
            self.vred = vred
            self.learning_rate = learning_rate
            self.save_hyperparameters(ignore=['vred'])
                                         
        def training_step(self, batch, batch_idx):
            features, labels = batch
            bag, tile = labels
            loss = self.vred.loss(features, tile)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.vred.parameters(), lr=self.learning_rate)
            return optimizer

    @dataclass
    class CONFIG:
        vred: VariationalReDecoder
        learning_rate: float = 1e-3

    @functools.cached_property
    def lightning_module(self):
        return self.Lightning(vred=self.cfg.vred, learning_rate=self.cfg.learning_rate)
    

class VariationalReDecoderStill(Datablock):
    VERSION = 1
    TOPICFILES = {'logs': None,
                  'ckpts': None,
    }
    
    @dataclass 
    class CONFIG:
        lightning: VariationalReDecoderLightning
        dataloader: torch.utils.data.DataLoader
        init_ckpt_path: str = None
        max_steps: int = 1

    def __init__(self, *args, n_devices: int = 1, **kwargs):
        super().__init__(*args, n_devices=n_devices, **kwargs)

    def valid(self):
        #TODO: check if max_steps has been run and a corresponding ckpt has been generated
        return False

    def __build__(self):
        logger = L.pytorch.loggers.TensorBoardLogger(save_dir=self.dirpath('logs'))
        self.log.debug(f"Built {logger=} for lightining {self.cfg.lightning}")
        self.log.debug(f"Building trainer for {self.cfg.max_steps=} for lightining {self.cfg.lightning}")
        trainer = L.pytorch.Trainer(
            default_root_dir=self.dirpath('ckpts'), 
            max_steps=self.cfg.max_steps,
            devices=self.n_devices,
            logger=logger,
        )
        self.log.debug(f"Built {trainer=} for lightining {self.cfg.lightning}")
        self.log.debug(f"Launching training for {self.cfg.max_steps=}")
        with torch.autograd.detect_anomaly(True):
            trainer.fit(model=self.cfg.lightning.lightning_module, train_dataloaders=self.cfg.dataloader)
        return self

  