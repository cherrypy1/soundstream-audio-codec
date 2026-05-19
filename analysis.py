from pathlib import Path
import random
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
import torch

from IPython.display import Audio, display
from params import take_params
from src.soundstream.utils.utils import get_model
from src.soundstream.dataset.audio_utils import load_audio, save_audio
from src.soundstream.losses import mel_spectrogram

ENGLISH_URLS = [
    "https://huggingface.co/datasets/mrfakename/noisy-speech-files/resolve/main/noisy_testset_wav/p232_007.wav",
    "https://huggingface.co/datasets/mrfakename/noisy-speech-files/resolve/main/noisy_testset_wav/p232_009.wav",
]
RUSSIAN_URLS = [
    "https://huggingface.co/datasets/niobures/russian-single-speaker-speech-dataset/resolve/main/early_short_stories/early_short_stories_0003.wav",
    "https://huggingface.co/bond005/whisper-large-v2-ru-podlodka/resolve/main/anna_matveeva_test.wav",
]


def load_model():
    config = take_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("checkpoints/final.ckpt", map_location=device)
    model, _ = get_model(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


def generate(model, audio, device):
    with torch.no_grad():
        return model(audio.unsqueeze(0).to(device), update_codebook=False)["audio"][
            0
        ].cpu()


def save_samples(real, fake, sample_rate, folder, index):
    folder = Path("outputs/analysis") / folder
    folder.mkdir(parents=True, exist_ok=True)

    real_path = folder / f"real_{index}.wav"
    fake_path = folder / f"fake_{index}.wav"
    save_audio(real_path, real, sample_rate)
    save_audio(fake_path, fake, sample_rate)
    return real_path, fake_path


def show_audio(real_path, fake_path):
    print("Real")
    display(Audio(str(real_path)))
    print("Reconstructed")
    display(Audio(str(fake_path)))


def plot_pair(real, fake, sample_rate):
    length = min(real.shape[-1], fake.shape[-1])
    real = real[..., :length].cpu()
    fake = fake[..., :length].cpu()
    time = torch.arange(length).numpy() / sample_rate

    window = torch.hann_window(1024)
    real_stft = (
        torch.stft(
            real.squeeze(0),
            n_fft=1024,
            hop_length=256,
            window=window,
            return_complex=True,
        )
        .abs()
        .clamp_min(1e-5)
        .log()
    )
    fake_stft = (
        torch.stft(
            fake.squeeze(0),
            n_fft=1024,
            hop_length=256,
            window=window,
            return_complex=True,
        )
        .abs()
        .clamp_min(1e-5)
        .log()
    )
    real_mel = mel_spectrogram(real.unsqueeze(0), 1024, 256, sample_rate)[0].log()
    fake_mel = mel_spectrogram(fake.unsqueeze(0), 1024, 256, sample_rate)[0].log()

    _, axes = plt.subplots(3, 2, figsize=(14, 9))

    axes[0, 0].plot(time, real.squeeze(0).numpy(), linewidth=0.5)
    axes[0, 0].set_title("Real waveform")
    axes[0, 0].set_xlabel("sec")

    axes[0, 1].plot(time, fake.squeeze(0).numpy(), linewidth=0.5)
    axes[0, 1].set_title("Fake waveform")
    axes[0, 1].set_xlabel("sec")

    axes[1, 0].imshow(real_mel.numpy(), origin="lower", aspect="auto")
    axes[1, 0].set_title("Real log Mel-spectrogram")

    axes[1, 1].imshow(fake_mel.numpy(), origin="lower", aspect="auto")
    axes[1, 1].set_title("Fake log Mel-spectrogram")

    axes[2, 0].imshow(real_stft.numpy(), origin="lower", aspect="auto")
    axes[2, 0].set_title("Real log STFT")

    axes[2, 1].imshow(fake_stft.numpy(), origin="lower", aspect="auto")
    axes[2, 1].set_title("Fake log STFT")

    plt.tight_layout()
    plt.show()


def get_audio_path(url, name):
    Path("outputs/analysis/downloads").mkdir(parents=True, exist_ok=True)
    path = Path("outputs/analysis/downloads") / f"{name}.wav"
    urlretrieve(url, path)
    return path


def show_qualitative_analysis(n_examples=2):
    config = take_params()
    model, device = load_model()
    files = sorted(Path(config["test_root"]).rglob("*.flac"))
    files = random.sample(files, min(n_examples, len(files)))

    for i, path in enumerate(files):
        real = load_audio(path, config["sample_rate"])
        fake = generate(model, real, device)
        real_path, fake_path = save_samples(
            real, fake, config["sample_rate"], "in_domain", i
        )

        show_audio(real_path, fake_path)
        plot_pair(real, fake, config["sample_rate"])


def show_external_english_analysis(urls=ENGLISH_URLS):
    config = take_params()
    model, device = load_model()

    for i, url in enumerate(urls):
        path = get_audio_path(url, f"external_english_{i}")

        real = load_audio(path, config["sample_rate"])
        fake = generate(model, real, device)
        real_path, fake_path = save_samples(
            real, fake, config["sample_rate"], "external_english", i
        )

        show_audio(real_path, fake_path)
        plot_pair(real, fake, config["sample_rate"])


def show_russian_analysis(urls=RUSSIAN_URLS):
    config = take_params()
    model, device = load_model()

    for i, url in enumerate(urls):
        path = get_audio_path(url, f"russian_{i}")

        real = load_audio(path, config["sample_rate"])
        fake = generate(model, real, device)
        real_path, fake_path = save_samples(
            real, fake, config["sample_rate"], "russian", i
        )

        show_audio(real_path, fake_path)
        plot_pair(real, fake, config["sample_rate"])


def audio_stats(real, fake, sample_rate):
    length = min(real.shape[-1], fake.shape[-1])
    real = real[..., :length]
    fake = fake[..., :length]

    real_mel = mel_spectrogram(real.unsqueeze(0), 1024, 256, sample_rate)
    fake_mel = mel_spectrogram(fake.unsqueeze(0), 1024, 256, sample_rate)

    return {
        "real_rms": real.pow(2).mean().sqrt().item(),
        "fake_rms": fake.pow(2).mean().sqrt().item(),
        "real_peak": real.abs().max().item(),
        "fake_peak": fake.abs().max().item(),
        "real_log_mel": real_mel.log().abs().mean().item(),
        "fake_log_mel": fake_mel.log().abs().mean().item(),
    }


def show_quantitative_statistics(n_files=100):
    config = take_params()
    model, device = load_model()
    files = sorted(Path(config["test_root"]).rglob("*.flac"))
    files = random.sample(files, min(n_files, len(files)))

    rows = []
    for path in files:
        real = load_audio(path, config["sample_rate"])
        fake = generate(model, real, device)

        row = audio_stats(real, fake, config["sample_rate"])

        row["rms_ratio"] = row["fake_rms"] / max(row["real_rms"], 1e-6)
        row["peak_ratio"] = row["fake_peak"] / max(row["real_peak"], 1e-6)
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = df.agg(["mean", "std", "min", "max"]).T
    display(summary)

    _, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(df["real_rms"], bins=20, alpha=0.6, label="real")
    axes[0, 0].hist(df["fake_rms"], bins=20, alpha=0.6, label="fake")
    axes[0, 0].set_title("RMS stats")
    axes[0, 0].legend()

    axes[0, 1].hist(df["real_peak"], bins=20, alpha=0.6, label="real")
    axes[0, 1].hist(df["fake_peak"], bins=20, alpha=0.6, label="fake")
    axes[0, 1].set_title("Peak stats")
    axes[0, 1].legend()

    axes[1, 0].hist(df["real_log_mel"], bins=20, alpha=0.6, label="real")
    axes[1, 0].hist(df["fake_log_mel"], bins=20, alpha=0.6, label="fake")
    axes[1, 0].set_title("Log Mel stats")
    axes[1, 0].legend()

    axes[1, 1].hist(df["rms_ratio"], bins=20, alpha=0.6, label="rms")
    axes[1, 1].hist(df["peak_ratio"], bins=20, alpha=0.6, label="peak")
    axes[1, 1].set_title("Fake-real ratios")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()
    return df
