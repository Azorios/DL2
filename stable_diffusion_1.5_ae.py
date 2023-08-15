import os
import PIL
import torch
import numpy as np
from PIL import Image
from einops import rearrange, repeat
from stablediffusion15.ldm.util import instantiate_from_config
from omegaconf import OmegaConf


def load_model_from_config(config, ckpt):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def main():
    input_dir = './output/imgs_cocoval/watermarked'
    output_dir = './output/imgs_cocoval/watermarked_removed'
    config_path = "configs/stable-diffusion/v1-inference.yaml"
    ckpt_path = "models/ldm/stable-diffusion-v1/model.ckpt"

    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            init_image = load_img(img_path).to(device)
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
            output_image = model.decode_first_stage(init_latent)
            output_image = torch.clamp((output_image + 1.0) / 2.0, min=0.0, max=1.0)
            output_image = 255. * rearrange(output_image.cpu().numpy(), '1 c h w -> h w c')
            Image.fromarray(output_image.astype(np.uint8)).save(os.path.join(output_dir, filename))

if __name__ == "__main__":
    main()
