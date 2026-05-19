import torch
import torch.optim as optim
from tqdm.auto import tqdm
from pathlib import Path

from src.soundstream.losses import discriminator_loss
from src.soundstream.utils.eval_metrics import evaluate_metrics
from src.soundstream.utils.metrics import count_losses

from src.soundstream.utils.utils import (
    set_random_seed,
    get_loader,
    get_model,
    get_exp,
    save_checkpoint,
)
from params import take_params


def one_batch_test(model, discriminator, train_loader, val_loader, config, device, exp):
    print("one batch test started", flush=True)
    model.train()
    discriminator.train()

    old_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    old_discr = {
        k: v.detach().cpu().clone() for k, v in discriminator.state_dict().items()
    }

    opt_model = optim.Adam(model.parameters(), lr=config["model_lr"], betas=(0.5, 0.9))
    opt_discr = optim.Adam(
        discriminator.parameters(), lr=config["discrim_lr"], betas=(0.5, 0.9)
    )

    one_batch_config = dict(config)
    one_batch_config["total_steps"] = config["one_batch_steps"]
    one_batch_config["log_every"] = 20

    fixed_batch = next(iter(train_loader))
    train_one_epoch(
        model,
        discriminator,
        train_loader,
        val_loader,
        opt_model,
        opt_discr,
        one_batch_config,
        device,
        0,
        float("inf"),
        exp=exp,
        fixed_batch=fixed_batch,
        log_prefix="one_batch",
    )

    model.load_state_dict(old_model)
    discriminator.load_state_dict(old_discr)
    print("one batch test finished")


def train_one_epoch(
    model,
    discriminator,
    train_loader,
    val_loader,
    opt_model,
    opt_discr,
    config,
    device,
    step,
    best_loss,
    exp=None,
    fixed_batch=None,
    log_prefix="train",
):
    model.train()
    discriminator.train()
    make_steps = 0
    checkpoint_dir = Path(config["checkpoint_dir"])

    if fixed_batch is None:
        loader = train_loader
    else:
        loader = [fixed_batch] * config["total_steps"]

    for batch in tqdm(loader):
        real = batch["audio"].to(device)
        with torch.no_grad():
            fake = model(real, update_codebook=False)["audio"]
        d_out_real = discriminator(real)
        d_out_fake = discriminator(fake.detach())
        d_loss = discriminator_loss(d_out_real, d_out_fake)

        opt_discr.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
        opt_discr.step()

        for p in discriminator.parameters():
            p.requires_grad_(False)
        losses, fake = count_losses(model, discriminator, real, config)
        model_loss = losses["model_loss"]

        opt_model.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt_model.step()
        for p in discriminator.parameters():
            p.requires_grad_(True)
        step += 1
        make_steps += 1

        if step >= config["total_steps"]:
            break

        if exp is not None and step % config["log_every"] == 0:
            train_metrics = {
                f"{log_prefix}_{name}": value.item() for name, value in losses.items()
            }
            train_metrics[f"{log_prefix}_discriminator_loss"] = d_loss.item()
            exp.log_metrics(train_metrics, step=step)

        if fixed_batch is None and step % config["save_every"] == 0:
            print(f"save checkpoint at step {step}", flush=True)
            save_checkpoint(
                model,
                discriminator,
                opt_model,
                opt_discr,
                config,
                step,
                checkpoint_dir / f"step_{step}.ckpt",
            )

        if fixed_batch is None and step % config["val_every"] == 0:
            val_metrics = evaluate_metrics(
                model, discriminator, val_loader, config, device, exp, step
            )
            model.train()
            discriminator.train()
            if val_metrics["val_rec_loss"] < best_loss:
                best_loss = val_metrics["val_rec_loss"]
                save_checkpoint(
                    model,
                    discriminator,
                    opt_model,
                    opt_discr,
                    config,
                    step,
                    checkpoint_dir / "best.ckpt",
                )
    return best_loss, step


def train():
    print("train started", flush=True)
    config = take_params()
    set_random_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)

    print("loading data...", flush=True)
    train_loader, val_loader = get_loader(config)
    print(
        f"train batches: {len(train_loader)}, val batches: {len(val_loader)}",
        flush=True,
    )
    model, discriminator = get_model(config, device)

    one_batch_exp = get_exp(config, "one_batch_test")
    one_batch_test(
        model, discriminator, train_loader, val_loader, config, device, one_batch_exp
    )

    opt_model = optim.Adam(model.parameters(), lr=config["model_lr"], betas=(0.5, 0.9))
    opt_discr = optim.Adam(
        discriminator.parameters(), lr=config["discrim_lr"], betas=(0.5, 0.9)
    )
    exp = get_exp(config, "train")

    checkpoint_dir = Path(config["checkpoint_dir"])
    best_loss = float("inf")
    step = 0

    print("training started", flush=True)
    while step < config["total_steps"]:
        best_loss, step = train_one_epoch(
            model,
            discriminator,
            train_loader,
            val_loader,
            opt_model,
            opt_discr,
            config,
            device,
            step,
            best_loss,
            exp,
        )

    print("save final checkpoint", flush=True)
    save_checkpoint(
        model,
        discriminator,
        opt_model,
        opt_discr,
        config,
        step,
        checkpoint_dir / "final.ckpt",
    )

    val_metrics = evaluate_metrics(
        model, discriminator, val_loader, config, device, exp, step
    )
    if val_metrics["val_rec_loss"] < best_loss:
        best_loss = val_metrics["val_rec_loss"]
        save_checkpoint(
            model,
            discriminator,
            opt_model,
            opt_discr,
            config,
            step,
            checkpoint_dir / "best.ckpt",
        )
    print("training finished", flush=True)


if __name__ == "__main__":
    train()
