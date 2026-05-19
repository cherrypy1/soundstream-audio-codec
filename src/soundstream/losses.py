import math

import torch
import torch.nn.functional as F
import torchaudio.functional as AF


def commitment_loss(encoder_output, quantized):
    return F.mse_loss(encoder_output, quantized.detach())


def discriminator_loss(real_outputs, fake_outputs):
    loss = 0
    for real, fake in zip(real_outputs, fake_outputs):
        real_logits = real[-1]
        fake_logits = fake[-1]
        loss = loss + F.relu(1 - real_logits).mean()
        loss = loss + F.relu(1 + fake_logits).mean()
    return loss / len(real_outputs)


def adversarial_loss(fake_outputs):
    loss = 0
    for fake in fake_outputs:
        fake_logits = fake[-1]
        loss = loss + F.relu(1 - fake_logits).mean()
    return loss / len(fake_outputs)


def feature_loss(real_outputs, fake_outputs):
    loss = 0
    count = 0
    for real, fake in zip(real_outputs, fake_outputs):
        for real_feature, fake_feature in zip(real[:-1], fake[:-1]):
            loss = loss + F.l1_loss(fake_feature, real_feature.detach())
            count += 1
    return loss / count


def mel_spectrogram(audio, n_fft, hop_length, sample_rate=16000):
    window = torch.hann_window(n_fft, device=audio.device)
    spec = torch.stft(
        audio.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()

    n_mels = 64
    if n_fft < 512:
        n_mels = n_fft // 8

    fb = AF.melscale_fbanks(
        n_fft // 2 + 1,
        0,
        sample_rate / 2,
        n_mels,
        sample_rate,
    ).to(audio.device)

    mel = spec.transpose(1, 2) @ fb
    return mel.transpose(1, 2).clamp_min(1e-5)


def reconstruction_loss(real, fake, sample_rate=16000):
    loss = real.new_tensor(0)
    for n_fft in [64, 128, 256, 512, 1024, 2048]:
        real_mel = mel_spectrogram(real, n_fft, n_fft // 4, sample_rate)
        fake_mel = mel_spectrogram(fake, n_fft, n_fft // 4, sample_rate)

        l1_loss = F.l1_loss(fake_mel, real_mel)
        log_diff = fake_mel.log() - real_mel.log()
        log_loss = torch.linalg.vector_norm(log_diff, dim=1).mean()
        loss = loss + l1_loss + math.sqrt(n_fft / 2) * log_loss

    return loss
