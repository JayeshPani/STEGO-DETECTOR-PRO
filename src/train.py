import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_model
from datasets import StegoDataset
from transforms import train_tfms, val_tfms
from utils import AverageMeter, compute_metrics


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, best_auc):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_auc": best_auc,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    # weights_only=False to allow loading older checkpoints safely (you created them)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", 0))
    best_auc = float(ckpt.get("best_auc", 0.0))
    return epoch, best_auc


def main(cfg):
    # Select device
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
    print("Device:", device)

    # Datasets / loaders
    pin_memory = device == "cuda"  # only helpful / supported on CUDA
    train_ds = StegoDataset(cfg["data"]["train_csv"], transform=train_tfms())
    val_ds = StegoDataset(cfg["data"]["val_csv"], transform=val_tfms())

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=pin_memory,
    )

    # Model / optim / sched / loss
    model = build_model(cfg["model"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    # Create scheduler (will be overwritten by checkpoint if resuming)
    total_epochs = int(cfg["train"]["epochs"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    criterion = nn.BCEWithLogitsLoss()

    amp_enabled = bool(cfg["train"]["use_amp"]) and (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if device == "cuda" else None

    # Resume logic
    start_epoch_cfg = int(cfg.get("start_epoch", 0))
    resume_path = cfg.get("resume", None)
    best_auc = 0.0
    start_epoch = 0

    if resume_path:
        try:
            start_epoch, best_auc = load_checkpoint(resume_path, model, optimizer, scheduler, scaler)
            print(f"[resume] loaded '{resume_path}' (epoch={start_epoch}, best_auc={best_auc:.4f})")
        except FileNotFoundError:
            print(f"[resume] file not found: {resume_path} (starting fresh)")
            start_epoch = start_epoch_cfg
        except Exception as e:
            print(f"[resume] failed to load '{resume_path}' ({type(e).__name__}: {e}) — starting fresh")
            start_epoch = start_epoch_cfg
    else:
        start_epoch = start_epoch_cfg

    # 🔁 Force LR override from config after resuming (ensures YAML LR is used)
    for g in optimizer.param_groups:
        g["lr"] = cfg["train"]["lr"]
    print(f"[info] Using learning rate from config: {cfg['train']['lr']}")

    # Training loop
    os.makedirs("outputs/checkpoints", exist_ok=True)
    last_ckpt = "outputs/checkpoints/efnb0_last.pth"
    best_ckpt = "outputs/checkpoints/efnb0_best.pth"

    for epoch in range(start_epoch + 1, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            loss_meter.update(loss.item(), x.size(0))
            pbar.set_postfix(loss=loss_meter.avg)

        if scheduler is not None:
            scheduler.step()

        # Save "last" checkpoint each epoch
        save_checkpoint(last_ckpt, epoch, model, optimizer, scheduler, scaler, best_auc)

        # Validate
        auc, f1 = compute_metrics(model, val_loader, device)
        print(f"Epoch {epoch}: val AUC={auc:.4f} F1={f1:.4f} (best={best_auc:.4f})")

        # Save "best"
        if auc > best_auc:
            best_auc = auc
            save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, scaler, best_auc)
            print(f"[best] improved AUC → {best_auc:.4f}; saved to {best_ckpt}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs("outputs/checkpoints", exist_ok=True)
    main(cfg)