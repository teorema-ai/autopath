"""
    > Background

        Contains code applying prov-gigapath foundation histopathology model to TCGA pancancer WSI data.
        The (prov-)gigapath model is described in https://www.nature.com/articles/s41586-024-07441-w
        Xu, H., Usuyama, N., Bagga, J. et al. "A whole-slide foundation model for digital pathology from real-world data.",
         Nature 630, 181–188 (2024). https://doi.org/10.1038/s41586-024-07441-w
        Code is available at https://github.com/prov-gigapath/prov-gigapath, 
        pretrained model can be obtained from HuggingFace: https://huggingface.co/prov-gigapath/prov-gigapath. 
"""

import argparse
from dataclasses import dataclass, asdict
import datetime
from functools import partial
import json
import logging
import math
import os
import pdb
import pickle
import sys
import time
import traceback as tb
from typing import List, Dict, Optional, Union, Tuple, Callable

import fsspec

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import scipy as sp


import ray
import torch
from torchvision import transforms

import timm

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from dinov2.models.vision_transformer import DinoVisionTransformer
#from dinov2.layers.attention import Attention
#from dinov2.layers import Mlp, PatchEmbed, NestedTensorBlock, DropPath

import dbx

from .augmentations import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, dino_tile_transform


def gigapath_tensor_block_class():
    from dinov2.layers import NestedTensorBlock
    class GigapathTensorBlock(NestedTensorBlock):
        def __init__(
            self,
            *,
            dim: int = 1536,
            num_heads: int = 24,
            init_values=1.0,
            drop_path: float,
            act_layer: Callable[..., nn.Module] = nn.SiLU,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_class: Callable[..., nn.Module] = None,
        ) -> None:
            from dinov2.layers.layer_scale import LayerScale

            if attn_class is None:
                from timm.models.vision_transformer import Attention
                attn_class = Attention
            nn.Module.__init__(self)
            # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
            qkv_bias = True
            proj_bias = True
            ffn_bias = True
            drop = 0.0
            attn_drop = 0.0

            self.norm1 = norm_layer(dim)
            self.attn = attn_class(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            from timm.layers.mlp import GluMlp
            from dinov2.layers import DropPath
            
            self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = 2*4096
            self.mlp = GluMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                bias=ffn_bias,
                gate_last=False,
            )
            self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

            self.sample_drop_ratio = drop_path
    return GigapathTensorBlock


class GigapathVisionTransformer(DinoVisionTransformer):
        def __init__(
            self, 
            *,
            drop_path_rate: float = 0.0,
            drop_path_uniform: bool = False,
            block_cls: Callable = None,
            attention_class: Callable[..., nn.Module] = None,
        ):
            from dinov2.models.vision_transformer import BlockChunk
            from dinov2.layers import PatchEmbed
            if block_cls is None:
                block_cls = gigapath_tensor_block_class()
            if attention_class is None:
                from timm.models.vision_transformer import Attention
                attention_class = Attention

            nn.Module.__init__(self)
            img_size = 224
            patch_size = 16
            in_chans = 3
            embed_dim = 1536
            depth = 40
            num_heads = 24
            drop_path_rate = drop_path_rate
            drop_path_uniform = drop_path_uniform
            init_values = 1.0  # for layerscale: None or 0 => no layerscale
            embed_layer = PatchEmbed
            act_layer = nn.SiLU
            block_chunks = 1
            num_register_tokens = 0
            interpolate_antialias = False
            interpolate_offset = 0.1

            norm_layer = partial(nn.LayerNorm, eps=1e-6)

            self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
            self.num_tokens = 1
            self.n_blocks = depth
            self.num_heads = num_heads
            self.patch_size = patch_size
            self.num_register_tokens = num_register_tokens
            self.interpolate_antialias = interpolate_antialias
            self.interpolate_offset = interpolate_offset

            self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
            assert num_register_tokens >= 0
            self.register_tokens = (
                nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
            )

            if drop_path_uniform is True:
                dpr = [drop_path_rate] * depth
            else:
                dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

            blocks_list = [
                block_cls(
                    dim=1536,
                    num_heads=24,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    attn_class=attention_class,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
            if block_chunks > 0:
                self.chunked_blocks = True
                chunked_blocks = []
                chunksize = depth // block_chunks
                for i in range(0, depth, chunksize):
                    # this is to keep the block index consistent if we chunk the block list
                    chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
                self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
            else:
                self.chunked_blocks = False
                self.blocks = nn.ModuleList(blocks_list)

            self.norm = norm_layer(embed_dim)
            self.head = nn.Identity()

            self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

            self.init_weights()

        def prepare_tokens_with_masks(self, x, masks=None):
            """
                For training we cannot use the timm.VisionTransformer code, 
                but must use the original dinov2 code.
            """
            '''
            #timm:
            x = self.patch_embed(x)
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
            return x
            '''
            B, nc, w, h = x.shape
            x = self.patch_embed(x)
            if masks is not None:
                x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.interpolate_pos_encoding(x, w, h)

            if self.register_tokens is not None:
                x = torch.cat(
                    (
                        x[:, :1],
                        self.register_tokens.expand(x.shape[0], -1, -1),
                        x[:, 1:],
                    ),
                    dim=1,
                )

            return x

            
def gigapath_vision_transformer_class():
    return GigapathVisionTransformer


def gigapath_tile_backbone(
    *, 
    type: str = 'dinov2', # or 'prov-gigapath'
    weights: str = None,
    cache: str = None,
    tile_encoder_snapshot: str = "8d2b1d2e65832e16bf9ff100a081acf6170a44ca",
    hf_token: str = "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", 
    device: str = 'cuda',
    resize: int = 256,
    center_crop: int = 224,
    drop_path_rate: float = 0.0,
    drop_path_uniform: bool = False,
    attention_class: Callable[..., nn.Module] = None,
    **kwargs,
):
    if attention_class is None:
        from dinov2.layers.attention import Attention
        attention_class = Attention
    os.environ['HF_TOKEN'] = hf_token
    if weights is None:
        if cache is None:
            cache = os.path.join(os.environ['HOME'], ".cache")
        weights = f"{cache}/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/{tile_encoder_snapshot}/pytorch_model.bin"
    
    if type == 'prov-gigapath':
        model = timm.create_model(
                "hf_hub:prov-gigapath/prov-gigapath", 
                pretrained=(weights is None)
            )
        if weights is not None:
            td = torch.load(weights, map_location=device)
            model.load_state_dict(td, strict=True)
    elif type == 'dinov2':
        layerscale = 1.0e-5
        GigapathVisionTransformer = gigapath_vision_transformer_class()
        model = GigapathVisionTransformer(
            drop_path_rate=drop_path_rate, 
            drop_path_uniform=drop_path_uniform,
            attention_class=attention_class,
        )
        state_dict = None
        if weights is not None:
            state_dict = torch.load(weights, map_location=device)
            def chkey(key):
                prefix = 'blocks.'
                if key.startswith(prefix): 
                    chkey = 'blocks.0.' + key[len(prefix):]
                else:
                    chkey = key
                return chkey
            state_dict_ = {chkey(key): val for key, val in state_dict.items()}
            state_dict_['mask_token'] = torch.zeros(1, model.embed_dim)
            model.load_state_dict(state_dict_, strict=True)
    else:
        raise ValueError(f"Uknown model type {type}")
    model.to(device)
    model.eval()
    return model


def backbone_blocks(model):
    if isinstance(model.blocks[0], torch.nn.modules.container.ModuleList):
        blocks = model.blocks[0]
    else:
        blocks = model.blocks
    return blocks


def gigapath_tile_backbone_with_sideband_and_preprocessor(
    *, 
    type: str = 'prov-gigapath',
    weights: str = None,
    cache: str = None,
    tile_encoder_snapshot: str = "8d2b1d2e65832e16bf9ff100a081acf6170a44ca",
    hf_token: str = "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", 
    device: str = 'cuda',
    resize: int = 256,
    center_crop: int = 224,
    **kwargs,
):
    model = gigapath_tile_backbone(
        type=type,
        weights=weights,
        cache=cache,
        tile_encoder_snapshot=tile_encoder_snapshot,
        hf_token=hf_token, 
        device=device,
        resize=resize,
        center_crop=center_crop,
        **kwargs,
    )

    # ---------------------------------------------------------------------
    num_features = 1536
    # This preprocessing, with resizing to 256 followed by
    # center crop to 224, is the same as the original Gigapath
    all_transforms = []

    if resize:
        all_transforms += [
            transforms.Resize(
                256 if resize is True else resize,
                interpolation=transforms.InterpolationMode.BICUBIC),
        ]
    if center_crop:
        all_transforms += [
            transforms.CenterCrop(
                224 if center_crop is True else center_crop),
        ]
    all_transforms += [
        transforms.Lambda(lambda x: x / 255.),
        transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD),
    ]
    transform = transforms.Compose(all_transforms)
    
    sideband = {}
    def capture_layer(name):
        def hook(model, input, output):
            sideband[f"{name}_input"] = input[0].detach()
            sideband[f"{name}"] = output.detach()
        return hook

    blocks = backbone_blocks(model)
    L = len(blocks)
    for l in range(L):
        blocks[l].norm1.register_forward_hook(capture_layer(f'B_{l}_norm1'))
        blocks[l].attn.qkv.register_forward_hook(capture_layer(f'B_{l}_attn_qkv'))
        blocks[l].attn.proj.register_forward_hook(capture_layer(f'B_{l}_attn_proj'))
        blocks[l].ls1.register_forward_hook(capture_layer(f'B_{l}_ls1'))
        blocks[l].norm2.register_forward_hook(capture_layer(f'B_{l}_norm2'))
        blocks[l].mlp.fc1.register_forward_hook(capture_layer(f'B_{l}_mlp_fc1'))
        blocks[l].mlp.act.register_forward_hook(capture_layer(f'B_{l}_mlp_act'))
        blocks[l].mlp.fc2.register_forward_hook(capture_layer(f'B_{l}_mlp_fc2'))
        blocks[l].mlp.drop1.register_forward_hook(capture_layer(f'B_{l}_mlp_drop1'))
        blocks[l].mlp.register_forward_hook(capture_layer(f'B_{l}_mlp'))
        blocks[l].mlp.register_forward_hook(capture_layer(f'B_{l}_ls2'))
    blocks[L-1].attn.q_norm.register_forward_hook(capture_layer(f'Q_{L-1}'))
    blocks[L-1].attn.k_norm.register_forward_hook(capture_layer(f'K_{L-1}'))
    model.patch_embed.register_forward_hook(capture_layer('patch_embed'))
    model.norm.register_forward_hook(capture_layer('norm'))
    model.head.register_forward_hook(capture_layer('head'))
    model.register_forward_hook(capture_layer('model'))
    return model, sideband, transform


def apply(backbone, sideband, transform, image, *, output_root: str = None, scale: bool = True):
    image_size = image.shape[1]
    timage = transform(image).to('cuda')
    output = backbone.cuda()(timage[None].cuda()).cpu().detach()
    timage = timage.cpu()
    sb = sideband

    blocks = backbone_blocks(backbone)
    L = len(blocks)
    attn = (sb[f'Q_{L-1}'].cpu()) @ (sb[f'K_{L-1}'].cpu().transpose(-2, -1))
    if scale:
        attn *= blocks[L-1].attn.scale 
    b, num_heads, num_patches_1, _ = attn.shape 
    map_size = int(np.sqrt(num_patches_1))
    
    attention_maps = {}
    for attention_head in range(num_heads):
        attention_map = attn[:,attention_head, 0, 1:]
        attention_map = attention_map.view(1, 1, map_size, map_size)
        attention_map = torch.nn.Upsample(size=(image_size, image_size))(attention_map)
        attention_map = attention_map[0, 0, :, :]
        attention_maps[(L-1, attention_head)] = attention_map.detach().cpu().numpy()
    if output_root is not None:
        os.makedirs(output_root, exist_ok=True)
        image_path = os.path.join(output_root, f"input.npz")
        with open(image_path, 'wb') as f:
            np.savez(f, image=image)
        output_path = os.path.join(output_root, f"output.npz")
        with open(output_path, 'wb') as f:
            np.savez(f, output=output)
        sideband_path = os.path.join(output_root, f"sideband.npz")
        sideband_ = dict(
            **{f"attention_map_{i}_{j}": attention_map for (i, j), attention_map in attention_maps.items()},
            **{k: v.cpu().detach().numpy() for k, v in sideband.items()},
        )
        with open(sideband_path, 'wb') as f:
            np.savez(f, **sideband_)
    return output, sideband_


class BackboneEvaluator:
    def __init__(self, 
        backbone=None,
        *,
        transform=None,
        device: str = 'cuda',
        log: dbx.Logger = dbx.Logger(stack_depth=3),
    ):
        self._backbone = backbone
        if self._backbone is None:
            self._backbone = "@autopath.gigaq.dinov2.backbone.gigapath_tile_backbone()"
        self.transform = transform
        if self.transform is None:
            self.transform = dino_tile_transform()
        self.device = device
        self.log = log

    @property
    def backbone(self):
        if isinstance(self._backbone, str):
            self.log.verbose(f"Evaluating {self._backbone}")
            self._backbone = dbx.eval_term(self._backbone).to(self.device)
        return self._backbone

    def to(self, device):
        self.device = device
        self._backbone = self.backbone.to(device)
        return self

    def eval(self):
        self.backbone.eval()
        return self

    def __call__(self, x):
        with torch.no_grad():
            y = self.transform(x.to(self.device))
            z = self.backbone(y).cpu().detach()
            del y
            return z


class SidebandBackboneEvaluator(BackboneEvaluator):

    def __init__(self, 
        backbone=None,
        *,
        transform=None,
        device: str = 'cuda',
        capture_blocks: Optional[List[int]] = None,
    ):
        super().__init__(backbone, transform=transform, device=device)
        self.capture_blocks = capture_blocks
        self.capture_layers = ['patch_embed', 'norm', 'head', 'norm', 'model',]
        self._sideband = None

    @property
    def sideband_layers(self):
        return self.capture_layers + (
            [] if self.capture_blocks is None else 
            [f"block.{b}" for b in self.capture_blocks]
        )

    @property
    def sideband(self):
        if self._sideband is None:
            self.log.verbose(f"Setting up sideband layer captures")
            self._sideband = {}
            def capture_layer(name):
                def hook(model, input, output):
                    self._sideband[f"{name}.input"] = input[0].cpu().detach()
                    self._sideband[f"{name}.output"] = output.cpu().detach()
                return hook
            
            blocks = backbone_blocks(self.backbone)
            if self.capture_blocks is not None:
                """
                blocks[l].norm1.register_forward_hook(capture_layer(f'block.{l}_norm1'))
                blocks[l].attn.qkv.register_forward_hook(capture_layer(f'block.{l}_attn_qkv'))
                blocks[l].attn.proj.register_forward_hook(capture_layer(f'block.{l}_attn_proj'))
                blocks[l].ls1.register_forward_hook(capture_layer(f'block.{l}_ls1'))
                blocks[l].norm2.register_forward_hook(capture_layer(f'block.{l}_norm2'))
                blocks[l].mlp.fc1.register_forward_hook(capture_layer(f'block.{l}_mlp_fc1'))
                blocks[l].mlp.act.register_forward_hook(capture_layer(f'block.{l}_mlp_act'))
                blocks[l].mlp.fc2.register_forward_hook(capture_layer(f'block.{l}_mlp_fc2'))
                blocks[l].mlp.drop1.register_forward_hook(capture_layer(f'block.{l}_mlp_drop1'))
                blocks[l].mlp.register_forward_hook(capture_layer(f'block.{l}_mlp'))
                blocks[l].mlp.register_forward_hook(capture_layer(f'block.{l}_ls2'))
                """
                for b in self.capture_blocks:
                    blocks[b].register_forward_hook(capture_layer(f'block.{b}'))
            for layer in self.capture_layers:
                getattr(self.backbone, layer).register_forward_hook(capture_layer(layer))
        return self._sideband
        