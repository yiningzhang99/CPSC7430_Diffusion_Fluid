"""Unconditional DDIM sampling from pure noise with a trained checkpoint (sanity check:
"do samples look like Kolmogorov-flow vortices?"). The original repo has no such entry point
(train_ddpm Diffusion.sample() is `pass`), so this reuses the original library code:
models/diffusion_new.Model and functions/denoising_step.ddim_steps.

Usage (from anywhere):
    python sample_uncond.py --lib ../Diffusion-based-Fluid-Super-resolution \
        --config configs/kmflow_re1000_rs64_sparse_recons_local.yml \
        --ckpt train_ddpm/experiments/km64_baseline/logs/km64_baseline/ckpt.pth \
        --data ../data/kf_2d_re1000_64_40seed.npy --n 8 --steps 100 --out samples_uncond.png
Paths for --config/--ckpt are relative to --lib.
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def dict2ns(d):
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, dict2ns(v) if isinstance(v, dict) else v)
    return ns


def main(a):
    data = np.load(os.path.abspath(a.data), mmap_mode="r")
    lib = os.path.abspath(a.lib)
    sys.path.insert(0, lib)
    os.chdir(lib)
    from models.diffusion_new import Model
    from functions.denoising_step import ddim_steps

    cfg = dict2ns(yaml.safe_load(open(a.config)))
    torch.manual_seed(a.seed)
    states = torch.load(a.ckpt, map_location="cuda")
    model = Model(cfg).cuda().eval()
    model.load_state_dict(states[-1] if a.weights == "ema" else states[0])
    print(f"loaded {a.weights} weights from {a.ckpt} (trained step {states[3]})")

    # data normalization = mean/std of the training seeds (same as runner / training)
    mean, std = float(np.mean(data[:-4, ::8])), float(np.std(data[:-4, ::8]))
    betas = torch.linspace(cfg.diffusion.beta_start, cfg.diffusion.beta_end,
                           cfg.diffusion.num_diffusion_timesteps).cuda()
    T = cfg.diffusion.num_diffusion_timesteps
    seq = range(0, T, T // a.steps)
    x = torch.randn(a.n, 3, cfg.data.image_size, cfg.data.image_size, device="cuda")
    xs, _ = ddim_steps(x, seq, model, betas)
    samples = (xs[-1] * std + mean)[:, 1].cpu().numpy()   # middle frame of each 3-frame triplet

    real = np.asarray(data[-1, np.linspace(0, data.shape[1] - 1, a.n).astype(int)])
    print(f"sample std {samples.std():.3f}  vs data std {real.std():.3f}")
    fig, ax = plt.subplots(2, a.n, figsize=(2 * a.n, 4.3))
    for i in range(a.n):
        ax[0, i].imshow(samples[i], cmap="twilight", vmin=-10, vmax=10)
        ax[1, i].imshow(real[i], cmap="twilight", vmin=-10, vmax=10)
        ax[0, i].axis("off"); ax[1, i].axis("off")
    ax[0, 0].set_title("DDIM samples", loc="left")
    ax[1, 0].set_title("real data (test seed)", loc="left")
    plt.tight_layout()
    plt.savefig(a.out, dpi=90)
    print("saved", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="samples_uncond.png")
    a = p.parse_args()
    a.out = os.path.abspath(a.out)
    main(a)
