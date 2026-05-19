import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.soundstream.codec import SoundStreamCodec
from src.soundstream.dataset.librispeech import LibriSpeechDataset
from src.soundstream.discriminator import Discriminator


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_loader(config):
    train_files = sorted(Path(config["train_root"]).rglob("*.flac"))

    test_files = sorted(Path(config["test_root"]).rglob("*.flac"))

    generator = torch.Generator().manual_seed(config["seed"])
    ids = torch.randperm(len(test_files), generator=generator).tolist()
    n_val = max(1, int(len(test_files) * config["val_percentage"]))
    val_files = [test_files[i] for i in ids[:n_val]]

    train_crop_samples = int(config["sample_rate"] * config["crop_seconds"])

    train_dataset = LibriSpeechDataset(
        train_files,
        config["sample_rate"],
        train_crop_samples,
        random_crop=True,
    )
    val_dataset = LibriSpeechDataset(
        val_files,
        config["sample_rate"],
        crop_samples=None,
        random_crop=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader


def get_model(config, device):
    model = SoundStreamCodec(
        sample_rate=config["sample_rate"],
        channels=config["channels"],
        latent_dim=config["latent_dim"],
        codebook_size=config["codebook_size"],
        num_quantizers=config["num_quantizers"],
    ).to(device)

    discriminator = Discriminator().to(device)
    return model, discriminator


def get_exp(config, name):

    api_key = os.getenv("COMET_API_KEY")
    if api_key is None:
        print("COMET_API_KEY is not set")
        return None

    import comet_ml

    exp = comet_ml.Experiment(
        api_key=api_key,
        project_name="bhw",
    )
    exp.set_name(name)
    exp.log_parameters(config)
    print("Exp started")
    return exp


def save_checkpoint(model, discriminator, opt_model, opt_discr, config, step, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_model": opt_model.state_dict(),
        "opt_discr": opt_discr.state_dict(),
        "config": config,
        "step": step,
    }
    torch.save(checkpoint, path)
