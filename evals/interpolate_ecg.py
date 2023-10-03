import torch
import imageio
import numpy as np
from matplotlib import pyplot as plt

from model.dcgan import Generator
from utils.dataloaders import set_seed
from evals.checkpoint_utils import load_checkpoint


def interpolate(noise1: torch.Tensor, noise2: torch.Tensor, num_interpolations: int) -> torch.Tensor:
    """
    Create interpolated noises between noise1 and noise2, including noise1 at the start and noise2 at the end.
    """
    alphas = torch.linspace(0, 1, num_interpolations + 2).to(noise1.device)  # +2 to account for noise1 and noise2
    noises = [(1 - alpha) * noise1 + alpha * noise2 for alpha in alphas]
    return torch.stack(noises, dim=0)


@torch.no_grad()
def save_interpolations(generator, noise1: torch.Tensor, noise2: torch.Tensor, num_interpolations: int, save_path: str):
    generator.eval()

    # create num_interpolation noises between noise1 and noise2
    noises = interpolate(noise1, noise2, num_interpolations)

    # generate fake data using the generator and noises
    fake_data = generator(noises).detach().cpu().numpy()

    # denormalize data from [-1, 1]
    global_max = [9.494109153747559, 7.599456787109375]
    global_min = [-10.515237808227539, -7.820725917816162]
    fake_data = (fake_data + 1) * (global_max[0] - global_min[0]) / 2 + global_min[0]

    # Visualize and save as gif
    images = []
    for i in range(len(fake_data)):
        fig, ax = plt.subplots(figsize=[10, 4], gridspec_kw={"hspace": 0})
        ax.plot(fake_data[i, 0, :])  # Plotting the data
        ax.set_title("Generated Data Visualization")

        # Convert Figure to image and append to images list
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)

    # Save images as gif
    imageio.mimsave(save_path, images, duration=250)
    print(f"Saved interpolation gif to {save_path}")


if __name__ == "__main__":
    checkpoint_path = (
        "checkpoints/ECG_GAN_2023_09_29_10_00_1.pt"  # change this to string path of checkpoint if you want to load from local
    )
    number_of_interpolations = 20
    save_path = "tmp/interpolation.gif"

    # set seed for reproducibility
    set_seed(23)

    # load checkpoint, config
    checkpoint, cfg = load_checkpoint(checkpoint_path)

    # Initialize generator
    generator = Generator(
        noise_size=cfg.generator.noise_size,
        output_size=cfg.data.size,
    ).to(cfg.system.device)

    generator.load_state_dict(checkpoint["generator_state_dict"])

    # create 2 noise vectors
    noise1 = torch.randn(cfg.generator.noise_size, cfg.data.channels, device=cfg.system.device)
    noise2 = torch.randn(cfg.generator.noise_size, cfg.data.channels, device=cfg.system.device)

    save_interpolations(generator, noise1, noise2, number_of_interpolations, save_path)
