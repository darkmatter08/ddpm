import logging
import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules import UNet
from utils import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cpu"):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_steps = noise_steps
        self.device = device
        self.img_size = img_size

        self.betas = self.prepare_noise_schedule()
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, -1)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)  # (1000,)
        self.sqrt_1_minus_alpha_bar = torch.sqrt(1-self.alpha_bar)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps, device=self.device)

    def noise_images(self, x, t):
        # x.shape = (12, 3, 64, 64)  (batch, RGB, img_size, img_size)
        noise = torch.randn_like(x, device=self.device)  # standard normal (gaussian) noise of shape x

        # unsqueeze self.*alpha_bar so that we can do an element-wise multiplication by x.
        part1 = self.sqrt_alpha_bar[t, None, None, None] * x
        part2 = self.sqrt_1_minus_alpha_bar[t, None, None, None] * noise
        x_t = part1 + part2  # noised_x

        # x_t.shape should be (12, 3, 64, 64)
        # print(f"{part1.shape=} {part2.shape=} {x_t.shape=}")

        return x_t, noise

    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps, (n, ), device=self.device)

    def sample(self, model, n):
        model.train(False)
        logging.info(f"Sampling {n} new images....")
        x_t = torch.randn(size=(n, 3, self.img_size, self.img_size), device=self.device)
        for t in tqdm(range(self.noise_steps - 1, 0, -1)):
            t = torch.Tensor((t,), device=self.device)
            t = t.int()

            if t > 1:
                z = torch.randn(size=(1, ))
            else:
                z = torch.zeros(size=(1, ))
            x_t = 1 / torch.sqrt(self.alphas[t]) * (x_t - ((self.betas[t] / self.sqrt_1_minus_alpha_bar[t]) * model(x_t, t))) + torch.sqrt(self.betas[t] * z)
            
        x_t = (x_t.clamp(-1, 1) + 1) / 2
        x_t = (x_t * 255).type(torch.uint8)
        return x_t


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    model.train(True)
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            # sample timesteps uniformly from [0, T]
            t = diffusion.sample_timesteps(args.batch_size)
            # noise the images
            x_t, noise = diffusion.noise_images(images, t)
            # predict the noise using the model
            predicted_noise = model(x_t, t)
            # calculate the loss
            loss = mse(predicted_noise, noise)
            # backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log the loss
            logger.add_scalar("Loss", loss.item(), epoch * l + i)
            pbar.set_description(f"Loss: {loss.item():.4f}")

        # TODO: save model.state_dict() at the end of every epoch
        torch.save(model.state_dict(), os.path.join(f"/Users/jains/code/Diffusion-Models-pytorch/models/{args.run_name}/ckpt_e{epoch}.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = "~/code/data/landscapes"
    args.device = "mps"
    args.lr = 3e-4
    train(args)


def sample():
    device = "cpu"
    model = UNet(device=device).to(device)
    ckpt = torch.load("/Users/jains/code/Diffusion-Models-pytorch/models/DDPM_Uncondtional/ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, 8)
    print(x.shape)
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in x.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


if __name__ == '__main__':
    # launch()
    sample()
