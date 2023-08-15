from diffusers import AutoencoderKL
model = AutoencoderKL.from_pretrained("kandinsky-community/kandinsky-2-2-prior", subfolder="prior")
