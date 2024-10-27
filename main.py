import torch
from architecture import Unet3D, GaussianDiffusion

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    num_frames = 8,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

fields = 'your input fields'
loss = diffusion(fields)
loss.backward()
# after training

sampled_fields = diffusion.sample(batch_size = 128)