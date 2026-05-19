from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF


def load_audio(path, sample_rate):
    audio, old_sample_rate = sf.read(str(path), dtype="float32")
    audio = torch.from_numpy(audio)

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.t()

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if old_sample_rate != sample_rate:
        audio = AF.resample(audio, old_sample_rate, sample_rate)

    return audio.contiguous()


def crop_audio(audio, size, random_crop=True):
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    length = audio.shape[-1]
    if length > size:
        max_start = length - size
        if random_crop:
            start = torch.randint(0, max_start + 1, (1,)).item()
        else:
            start = max_start // 2
        return audio[:, start : start + size]

    if length < size:
        return F.pad(audio, (0, size - length), mode="replicate")

    return audio


def save_audio(path, audio, sample_rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = audio.detach().cpu().clamp(-1, 1)
    if audio.ndim == 2:
        audio = audio.squeeze(0)

    sf.write(str(path), audio.numpy(), sample_rate)
