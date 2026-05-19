import torch
from torch import nn
from torch.nn.utils import weight_norm


class WaveDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(1, 16, 15, padding=7)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(16, 64, 41, stride=4, groups=4, padding=20)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(64, 256, 41, stride=4, groups=16, padding=20)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(256, 1024, 41, stride=4, groups=64, padding=20)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(1024, 1024, 41, stride=4, groups=256, padding=20)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(1024, 1024, 5, padding=2)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(1024, 1, 3, padding=1)),
        )

    def forward(self, x):
        features = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                features.append(x)
        return features


class MultiWaveDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [WaveDiscriminator() for _ in range(3)]
        )
        self.pool = nn.AvgPool1d(4, stride=2, padding=1)

    def forward(self, x):
        outputs = []
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                x = self.pool(x)
            outputs.append(discriminator(x))
        return outputs


class STFTResidualUnit(nn.Module):
    def __init__(self, channels, m, stride):
        super().__init__()
        self.skip = nn.Conv2d(channels, m * channels, 1, stride=stride)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                channels,
                m * channels,
                (stride[0] + 2, stride[1] + 2),
                stride=stride,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        y = self.net(x)
        skip = self.skip(x)
        a = min(y.shape[-2], skip.shape[-2])
        b = min(y.shape[-1], skip.shape[-1])
        return y[..., :a, :b] + skip[..., :a, :b]


class STFTDiscriminator(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(2, channels, 7, padding=3),
                STFTResidualUnit(channels, 2, (1, 2)),
                STFTResidualUnit(2 * channels, 2, (2, 2)),
                STFTResidualUnit(4 * channels, 1, (1, 2)),
                STFTResidualUnit(4 * channels, 2, (2, 2)),
                STFTResidualUnit(8 * channels, 1, (1, 2)),
                STFTResidualUnit(8 * channels, 2, (2, 2)),
                nn.Conv2d(16 * channels, 1, (1, 8)),
            ]
        )

    def forward(self, x):
        window = torch.hann_window(1024, device=x.device)
        x = torch.stft(
            x.squeeze(1),
            n_fft=1024,
            hop_length=256,
            window=window,
            return_complex=True,
        )
        x = x[:, :-1, :]
        x = torch.stack([x.real, x.imag], dim=1)
        x = x.permute(0, 1, 3, 2)

        features = []
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        return features


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = STFTDiscriminator()
        self.wave = MultiWaveDiscriminator()

    def forward(self, x):
        return [self.stft(x)] + self.wave(x)
