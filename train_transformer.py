import hydra
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from omegaconf import DictConfig

from vqgan.transformer import VQGANTransformer
from utils.dataloaders import create_dataloader


def configure_optimizers(model):
    decay, no_decay = set(), set()
    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    for mn, m in model.transformer.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn

            if pn.endswith("bias"):
                no_decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    no_decay.add("pos_emb")

    param_dict = {pn: p for pn, p in model.transformer.named_parameters()}

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
    return optimizer


def train_model(cfg: DictConfig, model: nn.Module, train_dataset, optim: torch.optim.Optimizer, epoch: int):
    with tqdm(range(len(train_dataset))) as pbar:
        for batch_idx, data in zip(pbar, train_dataset):
            optim.zero_grad()
            data = data.to(cfg.system.device)
            logits, targets = model(data)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optim.step()
            pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
            pbar.update(0)
            # TODO: wandb logging
    # TODO: save model and logs


@hydra.main(version_base=None, config_path="configs", config_name="config_transformer")
def main(cfg: DictConfig) -> None:
    # TODO wandb init

    model = VQGANTransformer(cfg).to(cfg.system.device)
    optim = configure_optimizers(model)
    train_loader = create_dataloader(cfg, seed=cfg.system.seed, splits=["train"])

    for epoch in range(1, cfg.epochs + 1):
        train_model(cfg, model, train_loader, optim, epoch)


if __name__ == "__main__":
    main()
