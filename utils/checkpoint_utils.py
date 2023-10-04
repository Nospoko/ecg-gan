import torch
from omegaconf import OmegaConf
from huggingface_hub.file_download import hf_hub_download


def load_checkpoint(ckpt_path: str = None, omegaconf: bool = True):
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
    else:
        checkpoint = torch.load(hf_hub_download("SneakyInsect/GANs", filename="double_generator_update_2023_09_26_21_23_3.pt"))
    cfg = checkpoint["config"]
    if omegaconf:
        cfg = OmegaConf.create(cfg)
    return checkpoint, cfg
