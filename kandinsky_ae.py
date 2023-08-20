import os

import torchvision.transforms as transforms
from Kandinsky2main.kandinsky2.kandinsky2_2_model import Kandinsky2_2, Image
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline, KandinskyV22Img2ImgPipeline, KandinskyV22InpaintPipeline
import torch



def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize the model
    model = Kandinsky2_2(device=device, task_type='img2img')

    # Directory containing the input images
    input_dir = './data/test/watermarked/0'

    # Directory to save the output images
    output_dir = 'kandinsky'
    os.makedirs(output_dir, exist_ok=True)

    # Transformation to apply to the images before processing
    transform = transforms.Compose([
        #transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        # Load and preprocess the image
        image = Image.open(os.path.join(input_dir, filename))
        image = transform(image).unsqueeze(0).to(device, dtype=torch.float16)
        print(image)

        # Process the image with the model
        #output = model.generate_img2img(prompt="", image=image)
        output = model.image_encoder(image)
        output = model.decoder(output)
        # Postprocess and save the output image

        output = output.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        output = (output * 255).astype('uint8')
        output = Image.fromarray(output)
        output.save(os.path.join(output_dir, filename))

def main2():
    from PIL import Image
    from diffusers import KandinskyV22Img2ImgPipeline
    from transformers import CLIPVisionModelWithProjection, AutoProcessor
    from diffusers.models import UNet2DConditionModel
    import torch

    device = torch.device("cpu")  # Change to "cuda" if you want to use a GPU

    # Load the CLIP processor and image encoder
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained('kandinsky-community/kandinsky-2-2-prior', subfolder='image_encoder').to(device)

    # Load the unet decoder and decoder pipeline
    unet = UNet2DConditionModel.from_pretrained('kandinsky-community/kandinsky-2-2-decoder', subfolder='unet').to(device)
    decoder = KandinskyV22Img2ImgPipeline.from_pretrained('kandinsky-community/kandinsky-2-2-decoder', unet=unet)

    # Open the image
    image = Image.open('./data/test/watermarked/0/00870002.png')

    # Process the image
    inputs = processor(images=image, return_tensors="pt")

    # Encode the image
    outputs = image_encoder(**inputs)
    print(outputs)
    print(outputs.image_embeds)

    # Decode the image and remove the watermark
    reconstructed_image = decoder(image_embeds= outputs.image_embeds, image=outputs.last_hidden_state, negative_image_embeds=outputs.image_embeds)
    print(reconstructed_image)

if __name__ == '__main__':
    main()

    #pip install huggingface_hub omegaconf transformers einops pytorch_lightning diffusers
    #pip install accelerate ?
    #pip install git+https://github.com/ai-forever/diffusers.git
