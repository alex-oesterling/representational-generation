#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import os, sys

from trainer import GenericTrainer
from trainer.image_utils import FaceFeatsModel, image_grid, plot_in_grid, expand_bbox, crop_face, image_pipeline, get_face_feats, get_face
import data_handler
import networks
from mpr.mpr import oracle_function
from mpr.mpr import getMPR

import itertools
import logging
import math
import shutil
import json
import pytz
import random
from datetime import datetime
from tqdm.auto import tqdm
import copy
import yaml
from pathlib import Path
import pickle
import wandb


import torch
from torch import nn
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import scipy

import kornia

import transformers
import diffusers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from accelerate.logging import get_logger

import diffusers
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import (
    LoraLoaderMixin,
)
from diffusers.models.attention_processor import (
    LoRAAttnProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.training_utils import EMAModel
from peft import LoraConfig

# you MUST import torch before insightface
# otherwise onnxruntime, used by FaceAnalysis, can only use CPU
from insightface.app import FaceAnalysis
from copy import deepcopy

my_timezone = pytz.timezone("Asia/Singapore")

os.environ["WANDB__SERVICE_WAIT"] = "300"  # set to DETAIL for runtime logging.

logger = get_logger(__name__)

class Trainer(GenericTrainer):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.modelname = self.args.functionclass
    
    def train(self, accelerator=None):
        # loader = data_handler.get_loader(self.args)
        loader = data_handler.DataloaderFactory.get_dataloader(dataname='fairface', args=self.args)
        model = self.model
        self.accelerator = accelerator
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model.name_or_path,
            subfolder="tokenizer"
            )
        self.text_encoder = model.text_encoder.to(self.accelerator.device)
        self.vae = model.vae.to(self.accelerator.device)
        self.unet = model.unet.to(self.accelerator.device)
        self.noise_scheduler = model.scheduler
        print('just ckecking the noise_scheduler:', self.noise_scheduler)

        self.vision_encoder = networks.ModelFactory.get_model(modelname='CLIP').to(self.accelerator.device)

        # Make linear projectors
        prompt = "a photo of person"
        prompts_token = self.tokenizer([prompt], return_tensors="pt", padding=True)
        prompts_token["input_ids"] = prompts_token["input_ids"].to(self.accelerator.device)
        prompts_token["attention_mask"] = prompts_token["attention_mask"].to(self.accelerator.device)

        prompt_embeds = self.text_encoder(
            prompts_token["input_ids"],
            prompts_token["attention_mask"],
        )
        prompt_embeds = prompt_embeds[0]
        print(prompt_embeds.shape)
        token_embedding_dim = prompt_embeds.shape[-1]
        self.linear_projector = nn.Linear(512, token_embedding_dim)
        
        # We only train the additional adapter LoRA layers
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.vision_encoder.requires_grad_(False)
        self.unet.enable_gradient_checkpointing()
        self.vae.enable_gradient_checkpointing()


        # For mixed precision training we cast all non-trainable weigths (self.vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype_high_precision = torch.float32
        self.weight_dtype = torch.float32
        if self.accelerator != None:
            if self.accelerator.mixed_precision == "fp16":
                self.weight_dtype = torch.float16
            elif self.accelerator.mixed_precision == "bf16":
                self.weight_dtype = torch.bfloat16

        # Move self.unet, self.vae and text_encoder to device and cast to self.weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        
        params_to_optimize = self.linear_projector.parameters()

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        
        # self-make a simple dataloader
        # the created train_dataloader_idxs should be identical across devices
        random.seed(self.args.seed+1)

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.args.max_train_steps * self.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )
        
        optimizer, lr_scheduler = self.accelerator.prepare(
                optimizer, lr_scheduler
            )

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num Iterations = {self.args.iterations}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        self.global_step = 0

        # Only show the progress bar once on each machine.
        self.wandb_tracker = self.accelerator.get_tracker("wandb", unwrap=True)
        
        for images in loader:
            iter += 1
            if iter > self.args.iterations:
                break
            images = images.to(self.accelerator.device, dtype=self.weight_dtype)
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            # Sample noise that we'll add to the latents   
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Generate image tokens
            image_embeds = self.vision_encoder.encode_image(images)
            image_token_embeds = self.linear_projector(image_embeds)
            image_token_embeds /= image_token_embeds.norm(dim=-1, keepdim=True)

            # Concatenate the image tokens with text tokens
            image_token_embeds = image_token_embeds.unsqueeze(1)
            text_image_embeds = torch.cat([prompt_embeds, image_token_embeds], dim=1)

            # Predict the noise residual and compute loss
            model_pred = self.unet(noisy_latents, timesteps, text_image_embeds, return_dict=False)[0]
            
            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
