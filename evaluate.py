import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from params import take_params
from src.soundstream.codec import SoundStreamCodec
from src.soundstream.dataset.audio_utils import save_audio
from src.soundstream.dataset.librispeech import LibriSpeechDataset
from src.soundstream.utils.eval_metrics import load_nisqa, load_stoi
from src.soundstream.utils.utils import get_exp, set_random_seed

N_AUDIO = 10


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    config = ckpt["config"]

    model = SoundStreamCodec(
        sample_rate=config["sample_rate"],
        channels=config["channels"],
        latent_dim=config["latent_dim"],
        codebook_size=config["codebook_size"],
        num_quantizers=config["num_quantizers"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def get_test_loader(config):
    dataset = LibriSpeechDataset(
        sorted(Path(config["test_root"]).rglob("*.flac")),
        sample_rate=config["sample_rate"],
        crop_samples=None,
        random_crop=False,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
    )


@torch.no_grad()
def evaluate_grade():
    config = take_params()
    set_random_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(config["checkpoint_dir"]) / "final.ckpt"

    model = load_model(checkpoint, device)
    loader = get_test_loader(config)
    stoi_metric = load_stoi(config["sample_rate"], device)
    nisqa_metric = load_nisqa(device)
    exp = get_exp(config, "evaluate_final")

    eval_dir = Path(config["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config["seed"])
    audio_ids = set(random.sample(range(len(loader)), min(N_AUDIO, len(loader))))

    stoi_values = []
    nisqa_values = []
    saved = 1

    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        real = batch["audio"].to(device)
        fake = model(real, update_codebook=False)["audio"]

        stoi_values.append(stoi_metric(fake.squeeze(1), real.squeeze(1)).item())
        nisqa_values.append(nisqa_metric(fake.squeeze(1))[0].item())

        if i in audio_ids:
            real_path = eval_dir / f"real_{saved}.wav"
            fake_path = eval_dir / f"fake_{saved}.wav"
            save_audio(real_path, real[0].cpu(), config["sample_rate"])
            save_audio(fake_path, fake[0].cpu(), config["sample_rate"])

            if exp is not None:
                exp.log_audio(
                    str(real_path),
                    sample_rate=config["sample_rate"],
                    file_name=f"real_{saved}.wav",
                )
                exp.log_audio(
                    str(fake_path),
                    sample_rate=config["sample_rate"],
                    file_name=f"fake_{saved}.wav",
                )
            saved += 1

    metrics = {
        "test_stoi": sum(stoi_values) / len(stoi_values),
        "test_nisqa_mos": sum(nisqa_values) / len(nisqa_values),
    }

    if exp is not None:
        exp.log_metrics(metrics)


if __name__ == "__main__":
    evaluate_grade()
