"""Downsample the Kolmogorov flow dataset (kf_2d_re1000_256_40seed.npy) by average pooling.

Usage:
    python scripts/downsample_kf.py data/kf_2d_re1000_256_40seed.npy data/kf_2d_re1000_64_40seed.npy --factor 4

Average pooling (rather than taking every 4th pixel) avoids aliasing of the small vortices.
The input is read with memory mapping and processed one slice of the first axis at a time,
so the 3.1 GB file never has to fit in RAM at once.
"""
import argparse

import numpy as np


def downsample(src: str, dst: str, factor: int) -> None:
    arr = np.load(src, mmap_mode="r")
    print(f"input shape: {arr.shape}, dtype: {arr.dtype}")

    h, w = arr.shape[-2:]
    if h % factor or w % factor:
        raise ValueError(f"spatial size {h}x{w} is not divisible by factor {factor}")
    out_shape = arr.shape[:-2] + (h // factor, w // factor)

    out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float32, shape=out_shape)
    for i in range(arr.shape[0]):
        block = np.asarray(arr[i], dtype=np.float32)
        pooled = block.reshape(block.shape[:-2] + (h // factor, factor, w // factor, factor))
        out[i] = pooled.mean(axis=(-3, -1))
        print(f"  slice {i + 1}/{arr.shape[0]} done")
    out.flush()
    print(f"output shape: {out_shape}, saved to {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="path to the 256x256 .npy file")
    parser.add_argument("dst", help="path for the downsampled .npy file")
    parser.add_argument("--factor", type=int, default=4, help="downsampling factor (default 4: 256 -> 64)")
    args = parser.parse_args()
    downsample(args.src, args.dst, args.factor)
