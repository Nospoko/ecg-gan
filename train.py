import io
import os

import hydra
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

import wandb
from utils.dataloaders import set_seed, create_dataloader
from model.dcgan import Generator, Discriminator, weights_init


def create_folders(cfg: DictConfig):
    logger_dict = cfg.logger

    for _, value in logger_dict.items():
        if value[-1] == "/":
            value = value[:-1]
            os.makedirs(value, exist_ok=True)


@torch.no_grad()
def visualize_training(generator: Generator, fixed_noise: torch.Tensor, epoch: int, batch_idx: int, chart_path: str = "tmp/"):
    generator.eval()

    # Generate fake data using the generator and fixed noise
    fake_data = generator(fixed_noise).detach().cpu().numpy()

    # Create a figure and a grid of subplots
    fig, axarr = plt.subplots(8, 2, figsize=(8, 16))

    # Loop through each subplot to plot the 2-channel data
    for i in range(8):
        for j in range(2):
            axarr[i, j].plot(fake_data[i, j, :])
            axarr[i, j].set_title(f"Noise {i+1}, Channel {j+1}")
            axarr[i, j].axis("off")
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])

    fig.suptitle(f"Epoch {epoch}, Batch {batch_idx}")
    fig.tight_layout()
    # save the figure to img/ folder
    fig.savefig(f"{chart_path}epoch_{epoch}_batch_{batch_idx}.png")
    return fig


def save_progress_gif(fig_list, chart_path: str = "tmp/"):
    imgs = []
    for fig in fig_list:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        imgs.append(img)

    # Save as an animated GIF
    imgs[0].save(f"{chart_path}progress.gif", save_all=True, append_images=imgs[1:], loop=0, duration=500)


def train_step(
    generator: Generator,
    discriminator: Discriminator,
    train_loader: DataLoader,
    gen_optimizer: torch.optim,
    disc_optimizer: torch.optim,
    criterion: nn.Module,
    cfg: DictConfig,
    fixed_noise: torch.Tensor,
    epoch: int,
) -> None:
    generator.train()
    discriminator.train()

    real_label = 1.0
    fake_label = 0.0

    fig_list = []

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, batch in progress_bar:
        real_data = batch["signal"].to(cfg.system.device)
        batch_size = real_data.size(0)

        # train discriminator
        disc_optimizer.zero_grad()
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=cfg.system.device)
        disc_real_output = discriminator(real_data).view(-1)
        discriminator_error_real = criterion(disc_real_output, label)
        discriminator_error_real.backward()
        D_x = disc_real_output.mean().item()  # Mean discriminator output for real data

        # train generator
        noise = torch.randn(batch_size, cfg.generator.nz, cfg.generator.output_channels, device=cfg.system.device)
        fake_data = generator(noise)
        label.fill_(fake_label)
        disc_fake_output = discriminator(fake_data.detach()).view(-1)

        discriminator_error_fake = criterion(disc_fake_output, label)
        discriminator_error_fake.backward()
        D_G_z1 = disc_fake_output.mean().item()  # Discriminator's average output when evaluating the fake data
        discriminator_error = discriminator_error_real + discriminator_error_fake
        disc_optimizer.step()
        generator.zero_grad()

        label.fill_(real_label)
        disc_output_after_update = discriminator(fake_data).view(-1)
        generator_error = criterion(disc_output_after_update, label)
        generator_error.backward()
        D_G_z2 = disc_output_after_update.mean().item()  # Discriminator's output after updating the generator, fake data
        gen_optimizer.step()

        # log to wandb
        wandb.log(
            {
                "generator_error": generator_error.item(),
                "discriminator_error": discriminator_error.item(),
                "D_x": D_x,
                "D_G_z1": D_G_z1,
                "D_G_z2": D_G_z2,
            }
        )
        if batch_idx % cfg.train.log_interval == 0:
            fig = visualize_training(generator, fixed_noise, epoch, batch_idx, cfg.logger.chart_path)
            fig_list.append(fig)

    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "gen_optimizer_state_dict": gen_optimizer.state_dict(),
        "disc_optimizer_state_dict": disc_optimizer.state_dict(),
        "config": cfg,
        "fixed_noise": fixed_noise,
    }
    torch.save(checkpoint, f"{cfg.logger.checkpoint_path}GAN_epoch_{epoch}.pth")
    return fig_list


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    name = f"DeterministicCheck_ECG_GAN_{cfg.run_date}"
    wandb.init(
        project="ECG GAN",
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    set_seed(cfg.system.seed)
    create_folders(cfg)

    # Initialize models
    discriminator_net = Discriminator(
        input_channels=cfg.discriminator.input_channels,
        input_size=cfg.discriminator.input_size,
        neurons=cfg.discriminator.neurons,
    ).to(cfg.system.device)

    generator_net = Generator(
        nz=cfg.generator.nz,
        output_channels=cfg.generator.output_channels,
        output_size=cfg.generator.output_size,
        neurons=cfg.generator.neurons,
    ).to(cfg.system.device)

    # Add random weights
    discriminator_net.apply(weights_init)
    generator_net.apply(weights_init)

    # criterion
    criterion = nn.BCELoss()
    # optimizer
    optimizer_discriminator = optim.Adam(
        discriminator_net.parameters(),
        lr=cfg.train.lr,
        betas=(cfg.discriminator.beta, 0.999),
    )
    optimizer_generator = optim.Adam(
        generator_net.parameters(),
        lr=cfg.train.lr,
        betas=(cfg.generator.beta, 0.999),
    )

    num_test_noises = 4
    # Fixed noise, used for visualizing training process
    fixed_noise = torch.randn(num_test_noises, cfg.generator.nz, cfg.generator.output_channels, device=cfg.system.device)
    # get loader:
    train_loader, _, _ = create_dataloader(cfg, seed=cfg.system.seed)

    all_figures = []
    # train epochs
    for epoch in range(1, cfg.train.epochs + 1):
        fig_list = train_step(
            generator=generator_net,
            discriminator=discriminator_net,
            train_loader=train_loader,
            gen_optimizer=optimizer_generator,
            disc_optimizer=optimizer_discriminator,
            criterion=criterion,
            cfg=cfg,
            fixed_noise=fixed_noise,
            epoch=epoch,
        )
        all_figures.extend(fig_list)
    # create gif from figures
    save_progress_gif(all_figures, cfg.logger.chart_path)


if __name__ == "__main__":
    main()
