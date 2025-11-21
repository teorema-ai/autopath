import gc
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import dbx

from .layers import UpLayer


class Classifier(nn.Module):
    def __init__(self, 
                 *, 
                 input_dim: int = 1536, 
                 hidden_dim: int = 512, 
                 n_hidden_layers: int = 1, 
                 n_hidden_activation_cls: Callable = nn.ReLU, 
                 n_classes=100,
                 log: dbx.Logger = dbx.Logger(),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.n_hidden_layers = n_hidden_layers
        self.log = log

        self.hidden_layers = nn.ModuleList()
        self.hidden_activations = nn.ModuleList()
        for i in range(n_hidden_layers):
            N = input_dim if i == 0 else hidden_dim
            self.hidden_layers.append(nn.Linear(N, hidden_dim))
            self.hidden_activations.append(n_hidden_activation_cls())
        self.last_layer = nn.Linear(hidden_dim, n_classes)

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
    

class ClassMultiscaleLatentGaussiansRGB(nn.Module):
    def __init__(self, 
                 *, 
                 n_classes: int = 100, 
                 input_dim: int = 1536, 
                 hidden_dim: int = 512, 
                 n_hidden_layers: int = 1, 
                 n_hidden_activation_cls: Callable = nn.ReLU, 
                 fine_scale: int = 256, 
                 n_scales: int = 5,
                 variance_eps: float = 0.01,
                 log: dbx.Logger = dbx.Logger(),
    ):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.fine_scale = fine_scale
        self.n_scales = n_scales
        self.variance_eps = variance_eps
        self.log = log

        self.latents = []
        self.means = []
        self.prevariances = []

        self.scales = [fine_scale//(4**i) for i in range(self.n_scales)]
        self.scale_dims = [3*scale**2 for scale in self.scales]
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

    def forward(self, x, c):
        #c shape (x.shape[0]()
        k = c[..., None]
        u = torch.cat([x, k], dim=-1) # a batch of [vector, scalar_class_idx]
        means_and_variances = []
        for i in range(len(self.latents)):
            w = self.latents[i](u)
            m = self.means[i](w)
            pv = self.prevariances[i](w)
            v = F.softplus(pv) + self.variance_eps
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
            setattr(self, f"up_layer_{layer_idx}", layer)
            in_channels = out_channels
        self.multiscale_resolutions = multiscale_resolutions or []
        self.variance_scale = variance_scale

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
    

class VariationalReDecoder(nn.Module):
    def __init__(self, 
                 *, 
                 classifier: Classifier, 
                 latent_gaussians: ClassMultiscaleLatentGaussiansRGB,
                 kernel_size: int = 3, 
                 use_batch_norm: bool = True,
                 variance_scale: float = 0.03,
                 class_batch_size: int = None,
                 log: dbx.Logger = dbx.Logger(),
    ):
        super().__init__()
        self.classifier = dbx.eval_term(classifier)
        self.latent_gaussians = dbx.eval_term(latent_gaussians)
        assert self.classifier.n_classes == self.latent_gaussians.n_classes, f"Classifier has {self.classifier.n_classes} classes but latent_gaussians has {self.latent_gaussians.n_classes} classes"
        self.decoder = ConvDecoder2D(
            num_input_features=latent_gaussians.n_channels,
            skip_features_per_layer=[latent_gaussians.n_channels]*self.latent_gaussians.n_scales,
            output_features_per_layer=[latent_gaussians.n_channels]*self.latent_gaussians.n_scales,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            variance_scale=variance_scale,
        )
        self.class_batch_size = class_batch_size
        self.log = log

    @property
    def n_classes(self):
        return self.classifier.n_classes
    
    def sample(self, x):
        self.log.debug(f"Computing class probabilities for x of shape: {x.shape}")
        class_probabilities = self.classifier(x)
        dist = torch.distributions.Categorical(probs=class_probabilities)
        ksample = dist.sample()
        self.log.debug(f"Sampled classes {ksample}")
        gaussian_means_and_variances = self.latent_gaussians(x, ksample)
        multiscale_means = []
        for mean, variance in gaussian_means_and_variances:
            normal_sample = torch.randn(mean.shape)
            multiscale_means.append(mean + normal_sample*torch.sqrt(variance))
        decoded_mean, decoded_variance = self.decoder(multiscale_means)
        normal_sample = torch.randn(decoded_mean.shape)
        decoded_sample = decoded_mean + normal_sample*torch.sqrt(decoded_variance)
        return decoded_sample

    def loss(self, x, y):
        classes = torch.tensor(list(range(self.n_classes))).reshape(1, -1).to(x.device)
        class_probabilities = self.classifier(x).reshape(1, -1)
        losses = []
        for class_lo in range(0, self.n_classes, self.class_batch_size):
            class_hi = min(self.n_classes, class_lo + self.class_batch_size)
            class_probabilities_batch = class_probabilities[:, class_lo:class_hi]
            classes_batch = classes[:, class_lo:class_hi]
            _loss = self._class_batch_loss(x, y, classes_batch, class_probabilities_batch)
            losses.append(_loss)
        loss = torch.sum(losses)
        del losses
        del class_probabilities
        del classes
        gc.collect()
        torch.cuda.empty_cache()
        return loss

    def _class_batch_loss(self, x, y, classes, class_probabilities):
        b = x.shape[0]
        k = classes.shape[1]
        C = classes.repeat(b, 1).T.reshape(b*k, 1)
        X = x.repeat(self.n_classes, [1]*len(x.shape[1:]))
        gaussian_means_and_variances = self.latent_gaussians(X, C)
        del X
        del C
        gc.collect()
        torch.cuda.empty_cache()
        multiscale_means = []
        for mean, variance in gaussian_means_and_variances:
            normal_sample = torch.randn(mean.shape)
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
        Y = y.repeat(self.n_classes, [1]*len(y.shape[1:]))
        _loss_ = F.sqrt((Mhat - Y)**2/Vhat) # (k b) c h w
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
    

class VariationalReDecoderEvaluator:
    def __init__(self, vred, dataloader, *, log: dbx.Logger = dbx.Logger()):
        self.vred = vred
        self.dataloader = dataloader
        self.log = log

    def sample(self, n_samples: int = 1, batch_size: int = 1):
        self.log.debug(f"Sampling {n_samples} samples of batch_size {batch_size}")
        output_batches_list = []
        sampleiter = iter(self.dataloader)
        for i in range(n_samples):
            input_batch = next(sampleiter)[0]
            output_batch = self.vred.sample(input_batch)
            output_batches_list.append(output_batch)
        return output_batches_list
