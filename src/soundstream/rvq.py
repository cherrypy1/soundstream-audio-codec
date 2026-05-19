import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, dim, decay=0.99):
        super().__init__()
        self.codebook_size = codebook_size
        self.decay = decay

        self.register_buffer("codebook", torch.randn(codebook_size, dim))
        self.register_buffer("usage", torch.zeros(codebook_size))
        self.register_buffer("avg", torch.zeros(codebook_size, dim))
        self.register_buffer("inited", torch.tensor(False))

    def get_nearest_ids(self, x):
        x_norm = x.pow(2).sum(dim=1, keepdim=True)
        codebook_norm = self.codebook.pow(2).sum(dim=1).unsqueeze(0)
        dist = x_norm - 2 * x @ self.codebook.t() + codebook_norm
        return dist.argmin(dim=1)

    @torch.no_grad()
    def init_codebook(self, x):
        ids = torch.randint(0, x.shape[0], (self.codebook_size,), device=x.device)
        self.codebook.copy_(x[ids])

        for _ in range(11):
            ids = self.get_nearest_ids(x)
            one_hot = F.one_hot(ids, self.codebook_size).type_as(x)
            count = one_hot.sum(dim=0)
            avg = one_hot.t() @ x

            used = count > 0
            self.codebook[used] = avg[used] / count[used].unsqueeze(1)

        ids = self.get_nearest_ids(x)
        one_hot = F.one_hot(ids, self.codebook_size).type_as(x)
        count = one_hot.sum(dim=0)

        self.usage.copy_(count)
        self.avg.copy_(self.codebook * self.usage.clamp_min(1.0).unsqueeze(1))
        self.replace_dead_codes(x)
        self.inited.fill_(True)

    @torch.no_grad()
    def update_codebook(self, x, ids):
        one_hot = F.one_hot(ids, self.codebook_size).type_as(x)
        count = one_hot.sum(dim=0)
        avg = one_hot.t() @ x

        self.usage.mul_(self.decay).add_(count, alpha=1 - self.decay)
        self.avg.mul_(self.decay).add_(avg, alpha=1 - self.decay)
        self.codebook.copy_(self.avg / self.usage.clamp_min(1e-6).unsqueeze(1))
        self.replace_dead_codes(x)

    @torch.no_grad()
    def replace_dead_codes(self, x):
        dead = self.usage < 2
        if not dead.any():
            return

        ids = torch.randint(0, x.shape[0], (int(dead.sum()),), device=x.device)
        self.codebook[dead] = x[ids]
        self.usage[dead] = 2
        self.avg[dead] = x[ids] * 2

    def forward(self, x, update_codebook=True):
        batch, channels, time = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, channels)

        if self.training and update_codebook and not bool(self.inited):
            self.init_codebook(flat)

        ids = self.get_nearest_ids(flat)
        quantized = self.codebook[ids]

        if self.training and update_codebook:
            self.update_codebook(flat, ids)

        quantized = quantized.reshape(batch, time, channels).permute(0, 2, 1)
        ids = ids.reshape(batch, time)
        return quantized, ids


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_quantizers, codebook_size, dim, decay=0.99):
        super().__init__()
        self.codebook_size = codebook_size
        self.layers = nn.ModuleList(
            [VectorQuantizer(codebook_size, dim, decay) for _ in range(num_quantizers)]
        )

    def forward(self, x, update_codebook=True):
        residual = x
        quantized = torch.zeros_like(x)
        all_ids = []

        for layer in self.layers:
            q, ids = layer(residual, update_codebook=update_codebook)
            quantized = quantized + q
            residual = residual - q.detach()
            all_ids.append(ids)

        quantized = x + (quantized - x).detach()

        return {
            "quantized": quantized,
            "perplexity": self.perplexity(all_ids, self.codebook_size),
        }

    @staticmethod
    def perplexity(all_ids, codebook_size):
        values = []
        for ids in all_ids:
            count = torch.bincount(ids.reshape(-1), minlength=codebook_size).float()
            p = count / count.sum().clamp_min(1)
            values.append(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))
        return torch.stack(values).mean()
