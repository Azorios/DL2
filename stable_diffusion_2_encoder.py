import argparse
import json
import os
import sys
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import utils
import utils_img
import utils_model

sys.path.append('src')
from src.ldm.models.autoencoder import AutoencoderKL
from src.ldm.models.diffusion.ddpm import LatentDiffusion
from src.loss.loss_provider import LossProvider
from torchvision.datasets import ImageFolder
def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    #aa("--train_dir", type=str, help="Path to the training data directory", required=True)
    #aa("--val_dir", type=str, help="Path to the validation data directory", required=True)

    group = parser.add_argument_group('Model parameters')
    aa("--ldm_config", type=str, default="test/v2-inference.yaml", help="Path to the configuration file for the LDM model")
    aa("--ldm_ckpt", type=str, default="test/v2-1_512-ema-pruned.ckpt", help="Path to the checkpoint file for the LDM model")
    aa("--msg_decoder_path", type=str, default= "/checkpoint/pfz/watermarking/models/hidden/dec_48b_whit.torchscript.pt", help="Path to the hidden decoder for the watermarking model")
    aa("--num_bits", type=int, default=48, help="Number of bits in the watermark")
    aa("--redundancy", type=int, default=1, help="Number of times the watermark is repeated to increase robustness")
    aa("--decoder_depth", type=int, default=8, help="Depth of the decoder in the watermarking model")
    aa("--decoder_channels", type=int, default=64, help="Number of channels in the decoder of the watermarking model")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=4, help="Batch size for training")
    aa("--img_size", type=int, default=256, help="Resize images to this size")
    aa("--loss_i", type=str, default="watson-vgg", help="Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.")
    aa("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
    aa("--lambda_i", type=float, default=0.2, help="Weight of the image loss in the total loss")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss in the total loss")
    aa("--optimizer", type=str, default="AdamW,lr=1e-4", help="Optimizer and learning rate for training")
    aa("--steps", type=int, default=100, help="Number of steps to train the model for")
    aa("--warmup_steps", type=int, default=20, help="Number of warmup steps for the optimizer")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=10, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=1000, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--num_keys", type=int, default=1, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")

    return parser

if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    import os
    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    # Directory with images to process
    input_dir = './output/imgs_cocoval/watermarked'
    # Directory to save processed images
    output_dir = './output/imgs_cocoval/watermarked_removed'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)  # Python 3.2 or newer

    # create device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load your model
    print(f'>>> Building LDM model with config {params.ldm_config} and weights from {params.ldm_ckpt}...')
    config = OmegaConf.load(f"{params.ldm_config}")
    ldm_ae: LatentDiffusion = utils_model.load_model_from_config(config, params.ldm_ckpt)
    ldm_ae: AutoencoderKL = ldm_ae.first_stage_model
    ldm_ae.eval()
    ldm_ae.to(device)

    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert PIL Image to PyTorch Tensor
    ])

    # walk through the input directory and process each file
    for root, dirs, files in os.walk(input_dir):

        for file in files:
            # check file extension, continue only if file is an image
            if file.endswith(('.png', '.jpg', '.jpeg')):
                # construct full file path
                file_path = os.path.join(root, file)

                # load the image and apply the transformation
                image = Image.open(file_path)
                image_tensor = transform(image)

                # add batch dimension
                img = image_tensor.unsqueeze(0).to(device)

                # apply the autoencoder
                imgs_z = ldm_ae.encode(img)  # b c h w -> b z h/f w/f
                imgs_z = imgs_z.mode()
                imgs_d0 = ldm_ae.decode(imgs_z)[0]  # b z h/f w/f -> b c h w

                # Convert torch tensor to PIL Image
                to_pil = transforms.ToPILImage()
                # Check if PIL is responsible for artifact

                image_pil = to_pil(imgs_d0.clamp(0,1).cpu())  # convert back to cpu before converting to PIL image

                # construct full output path
                output_path = os.path.join(output_dir, file)

                # save the processed image
                image_pil.save(output_path, format='PNG')
