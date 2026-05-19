import torch
from torch import nn

from src.soundstream.rvq import ResidualVectorQuantizer


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            2 * stride,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x[..., : -self.stride]


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            CausalConv1d(
                channels,
                channels,
                7,
                dilation=dilation,
            ),
            nn.ELU(),
            CausalConv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            ResidualUnit(channels // 2, 1),
            ResidualUnit(channels // 2, 3),
            ResidualUnit(channels // 2, 9),
            nn.ELU(),
            CausalConv1d(
                channels // 2,
                channels,
                2 * stride,
                stride=stride,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, channels: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(1, channels, 7),
            EncoderBlock(2 * channels, 2),
            EncoderBlock(4 * channels, 4),
            EncoderBlock(8 * channels, 5),
            EncoderBlock(16 * channels, 5),
            nn.ELU(),
            CausalConv1d(16 * channels, out_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            CausalConvTranspose1d(2 * channels, channels, stride),
            ResidualUnit(channels, 1),
            ResidualUnit(channels, 3),
            ResidualUnit(channels, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, channels: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(latent_dim, 16 * channels, 7),
            DecoderBlock(8 * channels, 5),
            DecoderBlock(4 * channels, 5),
            DecoderBlock(2 * channels, 4),
            DecoderBlock(channels, 2),
            nn.ELU(),
            CausalConv1d(channels, 1, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SoundStreamCodec(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 64,
        latent_dim: int = 128,
        codebook_size: int = 1024,
        num_quantizers: int = 8,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.strides = [2, 4, 5, 5]

        self.encoder = Encoder(channels, latent_dim)
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            dim=latent_dim,
        )
        self.decoder = Decoder(channels, latent_dim)

    def forward(
        self,
        audio: torch.Tensor,
        update_codebook: bool = True,
    ) -> dict[str, torch.Tensor]:
        z = self.encoder(audio)
        quantizer_out = self.quantizer(z, update_codebook=update_codebook)
        reconstructed = self.decoder(quantizer_out["quantized"])
        reconstructed = reconstructed[..., : audio.shape[-1]]
        return {
            "audio": reconstructed,
            "encoded": z,
            "quantized": quantizer_out["quantized"],
            "perplexity": quantizer_out["perplexity"],
        }
