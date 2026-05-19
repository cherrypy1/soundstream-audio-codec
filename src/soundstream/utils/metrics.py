from src.soundstream.losses import (
    adversarial_loss,
    commitment_loss,
    discriminator_loss,
    feature_loss,
    reconstruction_loss,
)


def count_losses(model, discriminator, real, config, validation=False):
    out = model(real)
    fake = out["audio"]

    real_outputs = discriminator(real)
    fake_outputs = discriminator(fake)

    rec_loss = reconstruction_loss(real, fake, config["sample_rate"])
    com_loss = commitment_loss(out["encoded"], out["quantized"])
    adv_loss = adversarial_loss(fake_outputs)
    feat_loss = feature_loss(real_outputs, fake_outputs)
    model_loss = rec_loss + com_loss + adv_loss + 100 * feat_loss

    losses = {
        "model_loss": model_loss,
        "rec_loss": rec_loss,
        "commitment_loss": com_loss,
        "adv_loss": adv_loss,
        "feature_loss": feat_loss,
        "perplexity": out["perplexity"],
    }
    if validation:
        losses["discriminator_loss"] = discriminator_loss(real_outputs, fake_outputs)
    return losses, fake


def add_metrics(total, metrics):
    for name, value in metrics.items():
        total[name] = total.get(name, 0) + value


def mean_metrics(total, n, prefix):
    result = {}
    for name, value in total.items():
        result[f"{prefix}_{name}"] = value / n
    return result
