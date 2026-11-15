"""Compare SDEdit reconstructions from the original runner with nearest / bicubic interpolation.

Reads the arrays the runner dumps into <run_dir>/sample_batch*/ (config `dump_arr: True`):
    reference_arr.npy            ground truth, middle frame of each triplet  (B, H, W)
    sample_arr_run_0_it0.npy     SDEdit output                               (B, H, W)
The low-res field is rebuilt from the ground truth exactly as in make_lowres_input.py
(f x f average pooling), so all methods see the same information:
    nearest : LR repeated f x f (this is also the SDEdit input)
    bicubic : periodic bicubic upsampling of LR (circular pad -> bicubic -> crop)

Metrics per frame, averaged: RMSE and relative L2 = ||x - gt|| / ||gt||.
Also plots the radially averaged kinetic-energy spectrum E(k) = 0.5 |w_hat|^2 / |k|^2.

Usage:
    python compare_recons.py <run_dir> --factor 4 --out_dir <dir>
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def avg_pool(x, f):
    b, h, w = x.shape
    return x.reshape(b, h // f, f, w // f, f).mean(axis=(2, 4))


def nearest_up(lr, f):
    return np.repeat(np.repeat(lr, f, axis=-2), f, axis=-1)


def bicubic_periodic(lr, f, pad=2):
    t = torch.from_numpy(lr)[:, None].double()
    t = F.pad(t, (pad, pad, pad, pad), mode="circular")
    up = F.interpolate(t, scale_factor=f, mode="bicubic", align_corners=False)
    c = pad * f
    return up[:, 0, c:-c, c:-c].numpy().astype(np.float32)


def metrics(x, gt):
    err = (x - gt).reshape(len(gt), -1)
    rmse = np.sqrt((err ** 2).mean(1))
    rel = np.linalg.norm(err, axis=1) / np.linalg.norm(gt.reshape(len(gt), -1), axis=1)
    return rmse.mean(), rmse.std(), rel.mean(), rel.std()


def energy_spectrum(w):
    n = w.shape[-1]
    wh = np.fft.fft2(w) / n ** 2
    k = np.fft.fftfreq(n, 1.0 / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0
    e = 0.5 * np.abs(wh) ** 2 / k2
    kr = np.sqrt(kx ** 2 + ky ** 2).round().astype(int)
    spec = np.array([e[..., kr == i].sum(-1).mean() for i in range(n // 2)])
    return np.arange(n // 2), spec


def main(a):
    dirs = sorted(glob.glob(os.path.join(a.run_dir, "sample_batch*")), key=lambda s: int(s.split("batch")[-1]))
    gt = np.concatenate([np.load(os.path.join(d, "reference_arr.npy")) for d in dirs]).astype(np.float32)
    sd = np.concatenate([np.load(os.path.join(d, f"sample_arr_run_0_it{a.it}.npy")) for d in dirs]).astype(np.float32)
    lr = avg_pool(gt, a.factor)
    methods = {"nearest": nearest_up(lr, a.factor), "bicubic": bicubic_periodic(lr, a.factor), "SDEdit": sd}

    os.makedirs(a.out_dir, exist_ok=True)
    lines = [f"run: {a.run_dir}", f"frames: {len(gt)}  ({gt.shape[-1]}x{gt.shape[-1]}, LR {lr.shape[-1]}x{lr.shape[-1]})",
             f"{'method':<10}{'RMSE':>16}{'rel L2':>18}"]
    for k, v in methods.items():
        r, rs, l, ls = metrics(v, gt)
        lines.append(f"{k:<10}{r:>9.4f} ± {rs:.4f}{l:>11.4f} ± {ls:.4f}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(a.out_dir, "metrics.txt"), "w", encoding="utf-8").write(txt + "\n")

    # example figure
    idx = np.linspace(0, len(gt) - 1, a.n_show).astype(int)
    cols = [("LR input (nearest)", methods["nearest"]), ("bicubic", methods["bicubic"]),
            ("SDEdit", methods["SDEdit"]), ("ground truth", gt)]
    fig, ax = plt.subplots(len(idx), 4, figsize=(9, 2.3 * len(idx)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(idx):
        for c, (name, arr) in enumerate(cols):
            ax[r, c].imshow(arr[i], cmap="twilight", vmin=-10, vmax=10)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(name, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, "compare_examples.png"), dpi=90)
    plt.close()

    # spectrum
    plt.figure(figsize=(5, 4))
    for name, arr, style in [("ground truth", gt, "k-"), ("bicubic", methods["bicubic"], "C0--"),
                             ("nearest", methods["nearest"], "C2:"), ("SDEdit", sd, "C3-")]:
        k, s = energy_spectrum(arr)
        plt.loglog(k[1:], s[1:], style, label=name)
    plt.axvline(lr.shape[-1] // 2, color="gray", lw=0.8)
    plt.text(lr.shape[-1] // 2 * 1.05, plt.ylim()[0] * 10, "LR Nyquist", color="gray", fontsize=8)
    plt.xlabel("wavenumber k"); plt.ylabel("E(k)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, "energy_spectrum.png"), dpi=90)
    print("saved figures to", a.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--factor", type=int, default=4)
    p.add_argument("--it", type=int, default=0, help="sample_step iteration to read")
    p.add_argument("--n_show", type=int, default=4)
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()
    a.out_dir = a.out_dir or os.path.join(a.run_dir, "comparison")
    main(a)
