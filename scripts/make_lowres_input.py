"""Build low-res SDEdit input files in the format the original runner expects.

The runner (runners/rs256_guided_diffusion.py::load_recons_data) reads `f[data_kw][-4:]`
from an .npz and expects the array to already be at the high-res size (N, T, H, W).
So we do:

    high-res (N, T, H, W) --downsample by `factor`--> (N, T, H/f, W/f)
                          --nearest upsample back--> (N, T, H, W)

Downsampling modes:
  subsample : take every f-th grid point (sparse measurements; used for the 256 -> 32 test,
              stand-in for the official figshare file kmflow_sampled_data_irregnew.npz, which
              samples *random* grid locations and could not be downloaded from the CLI)
  avg       : f x f average pooling (consistent with downsample_kf.py; used for 64 -> 16)

Only the last `n_test` seeds (the runner's test split) are stored.
Saved keys:  u{L}{L}_nearest, u{L}{L}  -> upsampled input (N_test, T, H, W)
             u{L}{L}_lr                -> the raw low-res field (N_test, T, L, L)

Usage:
    python make_lowres_input.py ../data/kf_2d_re1000_256_40seed.npy ../data/kmflow_lowres_u32.npz --factor 8
    python make_lowres_input.py ../data/kf_2d_re1000_64_40seed.npy  ../data/kmflow_lowres_u16_from64.npz --factor 4 --down avg
"""
import argparse

import numpy as np


def main(src, dst, factor, n_test, down):
    arr = np.load(src, mmap_mode="r")
    print(f"input shape: {arr.shape}")
    test = np.asarray(arr[-n_test:], dtype=np.float32)
    h, w = test.shape[-2:]
    if down == "subsample":
        low = test[..., ::factor, ::factor]
    elif down == "avg":
        low = test.reshape(test.shape[:-2] + (h // factor, factor, w // factor, factor)).mean(axis=(-3, -1))
    else:
        raise ValueError(down)
    up = np.repeat(np.repeat(low, factor, axis=-2), factor, axis=-1)
    assert up.shape == test.shape
    L = low.shape[-1]
    key = f"u{L}{L}"
    np.savez(dst, **{key + "_nearest": up, key: up, key + "_lr": low})
    print(f"saved keys [{key}_nearest, {key}, {key}_lr]; upsampled {up.shape}, low-res {low.shape} -> {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--factor", type=int, default=8)
    p.add_argument("--n_test", type=int, default=4)
    p.add_argument("--down", choices=["subsample", "avg"], default="subsample")
    a = p.parse_args()
    main(a.src, a.dst, a.factor, a.n_test, a.down)
