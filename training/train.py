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
from model.dcgan import Generator, Discriminator, weights_init


def create_folders(cfg: DictConfig):
    logger_dict = cfg.logger

    for _, value in logger_dict.items():
        if value[-1] == "/":
            value = value[:-1]
            os.makedirs(value, exist_ok=True)


@torch.no_grad()
def visualize_training(
    generator: Generator,
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
    for i in range(len(fixed_noise)):
        axes[i].plot(fake_data[i, 0, :])

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

    def random_labels(batch_size, start, end, device):
        return (end - start) * torch.rand(batch_size, device=device) + start

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, batch in progress_bar:
        real_data = batch[0].to(cfg.system.device)
        batch_size = real_data.size(0)

        # trick 6 from: https://github.com/soumith/ganhacks
        real_labels = random_labels(batch_size, 0.8, 1, cfg.system.device)
        fake_labels = random_labels(batch_size, 0.0, 0.1, cfg.system.device)

        # train discriminator
        discriminator.zero_grad()
        label = real_labels
        real_data = real_data.view(real_data.shape[0], 1, real_data.shape[1]).to(cfg.system.device)
        disc_real_output = discriminator(real_data).view(-1)
        discriminator_error_real = criterion(disc_real_output, label)
        discriminator_error_real.backward()
        D_x = disc_real_output.mean().item()  # Mean discriminator output for real data

        # train generator
        noise = torch.randn(batch_size, cfg.generator.noise_size, cfg.data.channels, device=cfg.system.device)
        fake_data = generator(noise)
        label = fake_labels
        disc_fake_output = discriminator(fake_data.detach()).view(-1)

        discriminator_error_fake = criterion(disc_fake_output, label)
        discriminator_error_fake.backward()
        D_G_z1 = disc_fake_output.mean().item()  # Discriminator's average output when evaluating the fake data
        discriminator_error = discriminator_error_real + discriminator_error_fake
        disc_optimizer.step()
        generator.zero_grad()

        label = real_labels
        disc_output_after_update = discriminator(fake_data).view(-1)
        generator_error = criterion(disc_output_after_update, label)
        generator_error.backward()
        D_G_z2 = disc_output_after_update.mean().item()  # Discriminator's output after updating the generator, fake data
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    run = wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        job_type="train",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    set_seed(cfg.system.seed)
    create_folders(cfg)

    # Initialize models
    discriminator = Discriminator(
        input_channels=cfg.data.channels,
        input_size=cfg.data.size,
        neurons=cfg.discriminator.neurons,
    ).to(cfg.system.device)

    generator = Generator(
        noise_size=cfg.generator.noise_size,
        output_size=cfg.data.size,
    ).to(cfg.system.device)

    # Add random weights
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    # criterion
    criterion = nn.BCELoss()
    # optimizer
    if cfg.train.use_sgd:
        optimizer_discriminator = optim.SGD(
            discriminator.parameters(),
            lr=cfg.train.discriminator_sgd_lr,
        )
    else:
        optimizer_discriminator = optim.Adam(
            discriminator.parameters(),
            lr=cfg.train.discriminator_adam_lr,
            betas=(cfg.discriminator.beta, 0.999),
        )

    optimizer_generator = optim.Adam(
        generator.parameters(),
        lr=cfg.train.generator_lr,
        betas=(cfg.generator.beta, 0.999),
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
        fixed_noise = torch.randn(num_test_noises, cfg.generator.noise_size, cfg.data.channels, device=cfg.system.device)
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
