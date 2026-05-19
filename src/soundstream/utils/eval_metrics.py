import torch
from tqdm.auto import tqdm

from src.soundstream.utils.metrics import add_metrics, count_losses, mean_metrics


def load_stoi(sample_rate, device):
    from torchmetrics.audio import ShortTimeObjectiveIntelligibility

    return ShortTimeObjectiveIntelligibility(sample_rate).to(device)


def load_nisqa(device):
    from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment

    return NonIntrusiveSpeechQualityAssessment(16000).to(device)


@torch.no_grad()
def evaluate_metrics(model, discriminator, val_loader, config, device, exp, step):
    model.eval()
    discriminator.eval()

    stoi_metric = load_stoi(config["sample_rate"], device)
    nisqa_metric = load_nisqa(device) if step % config["nisqa_every"] == 0 else None

    total = {}
    n = 0
    n_nisqa = 0
    real_sample = None
    fake_sample = None

    for batch in tqdm(val_loader):
        real = batch["audio"].to(device)
        losses, fake = count_losses(model, discriminator, real, config, validation=True)

        metrics = {name: value.item() for name, value in losses.items()}
        metrics["stoi"] = stoi_metric(fake.squeeze(1), real.squeeze(1)).item()
        add_metrics(total, metrics)
        n += 1

        if nisqa_metric is not None and n_nisqa < config["nisqa_batches"]:
            nisqa = nisqa_metric(fake.squeeze(1))
            total["nisqa_mos"] = total.get("nisqa_mos", 0) + nisqa[0].item()
            n_nisqa += 1

        if real_sample is None:
            real_sample = real[0].detach().cpu()
            fake_sample = fake[0].detach().cpu()

    metrics = mean_metrics(total, n, "val")
    if n_nisqa > 0:
        metrics["val_nisqa_mos"] = total["nisqa_mos"] / n_nisqa

    if exp is not None:
        exp.log_metrics(metrics, step=step)
        if real_sample is not None:
            exp.log_audio(
                real_sample.squeeze().numpy(),
                sample_rate=config["sample_rate"],
                file_name="val_real.wav",
                step=step,
            )
            exp.log_audio(
                fake_sample.squeeze().numpy(),
                sample_rate=config["sample_rate"],
                file_name="val_fake.wav",
                step=step,
            )
    return metrics
