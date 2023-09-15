import hydra
import torch
import wandb
import imageio
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

from utils.dataloaders import set_seed, create_dataloader
from model.dcgan import Generator, Discriminator, weights_init


@torch.no_grad()
def visualize_training(generator: Generator, fixed_noise: torch.Tensor, epoch: int, batch_idx: int):
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
    fig.savefig(f"img/epoch_{epoch}_batch_{batch_idx}.png")
    return fig


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
        output = discriminator(real_data).view(-1)
        discriminator_error_real = criterion(output, label)
        discriminator_error_real.backward()
        D_x = output.mean().item()

        # train generator
        noise = torch.randn(batch_size, cfg.generator.nz, cfg.generator.output_channels, device=cfg.system.device)
        fake_data = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake_data.detach()).view(-1)

        discriminator_error_fake = criterion(output, label)
        discriminator_error_fake.backward()
        D_G_z1 = output.mean().item()
        discriminator_error = discriminator_error_real + discriminator_error_fake
        disc_optimizer.step()
        generator.zero_grad()

        label.fill_(real_label)
        output = discriminator(fake_data).view(-1)
        generator_error = criterion(output, label)
        generator_error.backward()
        D_G_z2 = output.mean().item()
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
            fig = visualize_training(generator, fixed_noise, epoch, batch_idx)
            fig_list.append(fig)
    return fig_list


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    name = f"ECG_GAN_{cfg.run_date}"
    wandb.init(
        project="ECG GAN",
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    set_seed(cfg.system.seed)

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

    num_test_noises = 8
    # Fixed noise, used for visualizing training process
    fixed_noise = torch.randn(num_test_noises, cfg.generator.nz, cfg.generator.output_channels, device=cfg.system.device)
    # get loader:
    _, train_loader, _ = create_dataloader(cfg, seed=cfg.system.seed)

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
    imageio.mimsave(f"img/{name}.gif", all_figures, duration=1000)


if __name__ == "__main__":
    main()
