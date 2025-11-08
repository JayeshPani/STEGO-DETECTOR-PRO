# src/infer.py
import os
import sys
import glob
import json
import yaml
import argparse
import torch
import cv2
import numpy as np

# local imports
sys.path.append("src")
from transforms import val_tfms
from model import build_model


def _safe_load_state_dict(ckpt_path: str, map_location: str = "cpu"):
    """
    Loads either a full training checkpoint (with 'model' key) or a plain state_dict.
    """
    try:
        obj = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        # for older torch versions without weights_only
        obj = torch.load(ckpt_path, map_location=map_location)

    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def _device_str(prefer_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_batch_tensor(albumentations_image):
    """
    Accept both np.ndarray(HWC) and torch.Tensor(CHW) from Albumentations.
    Do NOT divide by 255 if Normalize()/ToTensorV2() already handled scaling.
    """
    if isinstance(albumentations_image, np.ndarray):
        # HWC -> CHW float32
        x = torch.from_numpy(albumentations_image.transpose(2, 0, 1)).float().unsqueeze(0)
    elif isinstance(albumentations_image, torch.Tensor):
        # CHW -> add batch
        x = albumentations_image.float().unsqueeze(0)
    else:
        raise TypeError(f"Unexpected transform output type: {type(albumentations_image)}")
    return x


def _load_model(cfg_path: str, ckpt_path: str, device: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # most repos expect cfg["model"] to contain model config
    model = build_model(cfg["model"]).to(device)
    state = _safe_load_state_dict(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_one(
    img_path: str,
    model: torch.nn.Module,
    device: str,
    temperature: float = 1.0,
    debug: bool = False,
):
    # Read image as BGR, convert to RGB
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Albumentations validation transforms (must match training!)
    tfm = val_tfms()
    out = tfm(image=rgb)
    img_tfm = out["image"]

    # To tensor (batch, C, H, W)
    x = _to_batch_tensor(img_tfm).to(device)

    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        # flatten to scalar
        logit = logits.view(-1)[0].item()
        # temperature scaling (T>1 -> softer, T<1 -> sharper)
        prob = torch.sigmoid(torch.tensor(logit / max(1e-8, temperature))).item()

    if debug:
        print(
            json.dumps(
                {
                    "image": img_path,
                    "device": device,
                    "logit": logit,
                    "prob": prob,
                    "temperature": temperature,
                },
                indent=2,
            )
        )

    return {"image": img_path, "logit": logit, "stego_prob": prob}


def main():
    p = argparse.ArgumentParser(description="EffNet-B0 Stego inference")
    p.add_argument("image_path", help="Path to an image, or a folder (will glob *.jpg *.png *.jpeg)")
    p.add_argument("--ckpt", default="outputs/checkpoints/efnb0_best.pth", help="Checkpoint path")
    p.add_argument("--cfg", default="configs/train_full.yaml", help="YAML config with model definition")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on probability")
    p.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling for logits")
    p.add_argument("--cpu", action="store_true", help="Force CPU (ignore CUDA/MPS)")
    p.add_argument("--no-mps", action="store_true", help="Disable MPS preference (Mac)")
    p.add_argument("--debug", action="store_true", help="Print logits etc.")
    args = p.parse_args()

    # device
    device = "cpu" if args.cpu else _device_str(prefer_mps=not args.no_mps)
    torch.set_grad_enabled(False)

    # model
    model = _load_model(args.cfg, args.ckpt, device)

    # images to run
    paths = []
    if os.path.isdir(args.image_path):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            paths.extend(glob.glob(os.path.join(args.image_path, ext)))
        paths.sort()
        if not paths:
            raise FileNotFoundError(f"No images found in folder: {args.image_path}")
    else:
        paths = [args.image_path]

    results = []
    for pth in paths:
        res = predict_one(
            pth,
            model=model,
            device=device,
            temperature=args.temperature,
            debug=args.debug,
        )
        res["threshold"] = args.threshold
        res["pred_label"] = int(res["stego_prob"] >= args.threshold)
        results.append(res)

    # print as lines of JSON (friendly for piping)
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()