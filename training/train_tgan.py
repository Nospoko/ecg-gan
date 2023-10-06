import os

import hydra
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

import wandb
from utils.dataloaders import set_seed, create_dataloader
from model.tgan import LSTMGenerator, CausalConvDiscriminator


def create_folders(cfg: DictConfig):
    logger_dict = cfg.logger

    for _, value in logger_dict.items():
        if value[-1] == "/":
            value = value[:-1]
            os.makedirs(value, exist_ok=True)


@torch.no_grad()
def visualize_training(
    generator,
    fixed_noise: torch.Tensor,
    epoch: int,
    batch_idx: int,
    chart_path: str = "tmp/",
):
    generator.eval()

    # Generate fake data using the generator and fixed noise
    fake_data = generator(fixed_noise).detach().cpu().numpy()

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=[10, 4],
        gridspec_kw={"hspace": 0},
    )

    # Loop through each subplot to plot the 2-channel data
    for i in range(fixed_noise.size(0)):
        axes[i].plot(fake_data[i, :, 0])

    fig.suptitle(f"Epoch {epoch}, Batch {batch_idx}")
    fig.tight_layout()
    # save the figure to chart_path/ folder
    # fig.savefig(f"{chart_path}epoch_{epoch}_batch_{batch_idx}.png")
    return fig


def average_gradient(model: nn.Module) -> dict:
    avg_gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            avg_gradients[name] = torch.mean(torch.abs(param.grad)).item()
    return avg_gradients


def train_epoch(
    generator,
    discriminator,
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

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, batch in progress_bar:
        ###########################
        # Update Discriminator
        ###########################

        # Train with real data
        discriminator.zero_grad()
        real_data = batch[0].to(cfg.system.device).unsqueeze(2)
        batch_size, seq_len = real_data.size(0), real_data.size(1)
        label = torch.full((batch_size, seq_len, 1), real_label, device=cfg.system.device)

        real_output = discriminator(real_data)
        errD_real = criterion(real_output, label)
        errD_real.backward()
        D_x = real_output.mean().item()

        # Train with fake data
        noise = torch.randn(batch_size, seq_len, cfg.generator.noise_size, device=cfg.system.device)
        fake_data = generator(noise)
        label.fill_(fake_label)
        fake_output = discriminator(fake_data.detach())
        errD_fake = criterion(fake_output, label)
        errD_fake.backward()
        D_G_z1 = fake_output.mean().item()
        discriminator_error = errD_real + errD_fake
        disc_optimizer.step()

        ###########################
        # Update Generator
        ###########################

        generator.zero_grad()
        label.fill_(real_label)
        fake_output = discriminator(fake_data)
        generator_error = criterion(fake_output, label)
        generator_error.backward()
        D_G_z2 = fake_output.mean().item()
        gen_optimizer.step()

        # log to wandb
        if batch_idx % cfg.train.log_interval == 0:
            generator_gradients = average_gradient(generator)
            wandb.log(
                {
                    "generator_error": generator_error.item(),
                    "discriminator_error": discriminator_error.item(),
                    "D_x": D_x,
                    "D_G_z1": D_G_z1,
                    "D_G_z2": D_G_z2,
                    "generator/generator_gradients": generator_gradients,
                },
                commit=False,
            )
            fig = visualize_training(generator, fixed_noise, epoch, batch_idx, cfg.logger.chart_path)
            generator.train()
            wandb.log({"fixed noise": wandb.Image(fig)}, commit=True)
            # TODO: Might add local saving here as well
            plt.close(fig)

    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "gen_optimizer_state_dict": gen_optimizer.state_dict(),
        "disc_optimizer_state_dict": disc_optimizer.state_dict(),
        "config": OmegaConf.to_object(cfg),
        "fixed_noise": fixed_noise,
    }
    torch.save(checkpoint, f"{cfg.logger.checkpoint_path}{cfg.run_name}_{epoch}.pt")


@hydra.main(version_base=None, config_path="../configs", config_name="config_tgan")
def main(cfg: DictConfig):
    run = wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        job_type="train",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    set_seed(cfg.system.seed, deterministic=False)
    create_folders(cfg)

    # Initialize models
    generator = LSTMGenerator(
        in_dim=cfg.generator.noise_size,
        out_dim=cfg.data.channels,
        n_layers=cfg.generator.n_layers,
        hidden_dim=cfg.generator.hidden_dim,
    ).to(cfg.system.device)
    discriminator = CausalConvDiscriminator(
        input_size=cfg.data.channels,
        n_layers=cfg.discriminator.n_layers,
        n_channel=cfg.discriminator.n_channel,
        kernel_size=cfg.discriminator.kernel_size,
        dropout=cfg.discriminator.dropout,
    ).to(cfg.system.device)

    # criterion
    criterion = nn.BCELoss().to(cfg.system.device)
    # optimizer
    optimizer_discriminator = optim.Adam(
        discriminator.parameters(),
        lr=cfg.train.discriminator_lr,
    )

    optimizer_generator = optim.Adam(
        generator.parameters(),
        lr=cfg.train.generator_lr,
    )

    if cfg.train.load_checkpoint is not None:
        checkpoint = torch.load(cfg.train.load_checkpoint)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_generator.load_state_dict(checkpoint["gen_optimizer_state_dict"])
        optimizer_discriminator.load_state_dict(checkpoint["disc_optimizer_state_dict"])
        fixed_noise = checkpoint["fixed_noise"]
        epoch = checkpoint["epoch"]
    else:
        num_test_noises = 4
        epoch = 0
        # Fixed noise, used for visualizing training process
        fixed_noise = torch.randn(num_test_noises, cfg.data.size, cfg.generator.noise_size, device=cfg.system.device)

    # get loader:
    # train_loader, _, _ = create_dataloader(cfg, seed=cfg.system.seed)
    train_loader = create_dataloader(cfg, seed=cfg.system.seed, splits=["train", "test", "validation"])
    print(len(train_loader))

    # train epochs
    epochs = cfg.train.epochs if epoch == 0 else cfg.train.more_epochs
    start_epoch = epoch + 1
    for epoch in range(start_epoch, epochs + 1):
        train_epoch(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            gen_optimizer=optimizer_generator,
            disc_optimizer=optimizer_discriminator,
            criterion=criterion,
            cfg=cfg,
            fixed_noise=fixed_noise,
            epoch=epoch,
        )
    run.finish()


if __name__ == "__main__":
    main()
