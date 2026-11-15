# Reproduction log: Shu, Li & Barati Farimani (2023), physics-informed diffusion for flow reconstruction

Goal of this phase: get the original repo (`Diffusion-based-Fluid-Super-resolution/`, branch `main_v1`)
running locally, not yet writing our own implementation. Every code change in the original repo is
marked with a `# [local]` comment so `grep -rn "\[local\]"` finds them all.

Machine: Windows 11, RTX 2060 Max-Q (6 GB, sm_75), driver 555.97, 23 GB RAM.

## Layout

```
./
├── data/                                  # not in git (see README for download); generated files marked [new]
│   ├── kf_2d_re1000_256_40seed.npy        # original HR data, (40 seeds, 320 frames, 256, 256) float32, 3.35 GB
│   ├── kf_2d_re1000_64_40seed.npy         # [new] 4x avg-pooled, (40, 320, 64, 64)
│   ├── kmflow_lowres_u32.npz              # [new] 256 -> 32 (subsample) -> 256 (nearest), test seeds only
│   └── kmflow_lowres_u16_from64.npz       # [new] 64 -> 16 (avg pool) -> 64 (nearest), test seeds only
├── scripts/
│   ├── downsample_kf.py                   # 256 -> 64 avg pooling
│   ├── make_lowres_input.py               # builds SDEdit input .npz in the runner's format
│   ├── sample_uncond.py                   # unconditional DDIM sampling sanity check
│   └── compare_recons.py                  # SDEdit vs nearest vs bicubic metrics + figures
├── results/                               # figures, metrics, training log
└── Diffusion-based-Fluid-Super-resolution/   # original repo (modified, see below)
```

---

## Step 1: environment

README says: python 3.8, PyTorch 1.7 + CUDA 10.1, torchvision 0.8.2, TensorBoard 2.11, NumPy 1.22,
tqdm 4.59, einops 0.4.1, matplotlib 3.6.2. There is no requirements.txt.

No conda was installed, so Miniconda was installed in the user folder
(JustMe, not added to PATH, not registered as the system Python).

```bash
conda create -n pidm python=3.8 pip -c conda-forge --override-channels -y   # conda-forge avoids the defaults-channel ToS prompt
conda activate pidm
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.22.4 tensorboard==2.11.0 tqdm==4.59.0 einops==0.4.1 matplotlib==3.6.2 \
            pyyaml scikit-learn==1.1.3 tensorboardX==2.5.1
```

- torch pulled numpy 1.24, so it was re-pinned to 1.22.4 as in the README.
- **Not listed in the README but imported by `train_ddpm/`:** `scikit-learn` (`datasets/utils.py`), `tensorboardX`
  (`runners/diffusion_tub.py`), `pyyaml`.

Check:
```
torch 1.7.1+cu101 cuda 10.1 cudnn 7604
cuda available: True
NVIDIA GeForce RTX 2060 with Max-Q Design capability (7, 5) VRAM 6.00 GiB
```

---

## Step 2: 256x256 inference with the pretrained checkpoint (batch 1)

Pretrained weights (README links): `pretrained_weights/baseline_ckpt.pth` (unconditional) and
`conditional_ckpt_new.pth` (it had been saved in `configs/` and should be moved to `pretrained_weights/` before using it).

### Problems with the README instructions, and fixes

| # | Problem | Cause | Fix |
|---|---|---|---|
| 2.1 | README command `--config kmflow_re1000_rs256.yml` cannot work | `reconstruct()` reads `data.sample_data_dir`, `data_kw`, `smoothing`, `sampling.log_loss`, `dump_arr`; only the `*_sparse_recons*.yml` configs define them | Use `kmflow_re1000_rs256_sparse_recons.yml` as the base |
| 2.2 | Low-res input `kmflow_sampled_data_irregnew.npz` missing | figshare (file 39214622) answers CLI downloads with HTTP 202 + empty body (bot check) | `scripts/make_lowres_input.py` builds a stand-in in the same format (see below) |
| 2.3 | Data path | data lives in `data/` next to the library folder, configs expect `./data/` inside the repo | New config uses `../data/...` |
| 2.4 | `IndexError: list index out of range` in `make_image_grid` | grid rows = `t // ncols` with `ncols=10`; with batch 1 that is 0 rows; the original batch 20 hid it | `runners/rs256_guided_diffusion.py`: `ncols = min(ncols, t)`, `rows = ceil(t / ncols)` |
| 2.5 | `AttributeError: 'builtin_function_or_method' object has no attribute 'fft2'` | In PyTorch 1.7, `torch.fft` is still the legacy *function*; the `torch.fft` module only replaces it after `import torch.fft` | Add `import torch.fft` to the 4 files using it |
| 2.6 | `AttributeError: module 'torch.fft' has no attribute 'fft2'` | **`fft2/rfft2/irfft2` were added in PyTorch 1.8**; 1.7.1 only has `fftn/rfftn/irfftn`. The code cannot have run on 1.7 as written (README's "tested on PyTorch 1.7" is inaccurate) | `fft2(x, dim=[2,3])` -> `fftn(x, dim=[2,3])`, `irfft2(.., dim=[2,3])` -> `irfftn(.., dim=[2,3])`, `rfft2(x)` -> `rfftn(x, dim=(-2,-1))`. Verified identical to `numpy.fft.fft2`. (Alternative: torch 1.8.1+cu101, which needs no change.) |
| 2.7 | Full test set = 4 seeds x 318 triplets = 1272 samples; batch 1 would take hours | n/a | `--max_batches N` CLI option in `main.py` + early `break` in the runner; final mean only over processed samples |
| 2.8 | `num_workers: 4` | On Windows DataLoader workers are spawned, and each would get a pickled copy of the ~2 GB test tensors | `num_workers: 0` in local configs |
| 2.9 | Train/inference architecture mismatch (found in step 4, see 4.5) | Inference `models/diffusion_new.py` `Model.conv_out` uses zero padding, training `train_ddpm/models/diffusion.py` uses `padding_mode='circular'` | Config option `model.conv_out_padding` (default `zeros` = original behaviour); set to `circular` in both local configs. **The pretrained 256 checkpoint was also trained with circular padding**: with it, eps-MSE on the border falls ~10x (0.080 -> 0.005) and overall by 20-30% |

Warning `Casting complex values to real discards the imaginary part` comes from the backward of the complex
FFT in `voriticity_residual`. Checked: the autograd gradient matches central finite differences to 6-7
significant digits, so the warning is harmless (the input is real, so dropping the imaginary part of its gradient is correct).

**Stand-in low-res input**: `make_lowres_input.py --factor 8` takes every 8th grid point (32x32) of the 4 test
seeds and nearest-upsamples back to 256, saved under keys `u3232_nearest` / `u3232`. It differs from the official file,
which samples *random* grid locations ("irreg"); if that file is downloaded via a browser it can be dropped in unchanged.

### Files changed / added

- `configs/kmflow_re1000_rs256_sparse_recons_local.yml` (new; copy of `..._sparse_recons.yml`): data paths, `batch_size: 1`, `num_workers: 0`, own `log_dir`.
- `main.py`: `--max_batches`.
- `runners/rs256_guided_diffusion.py`: `import torch.fft`; `fft2/irfft2` -> `fftn/irfftn`; `make_image_grid` row count; `--max_batches` early stop and averaging over processed samples.
- `models/diffusion_new.py`, `train_ddpm/models/diffusion.py`: `import torch.fft`; FNO `rfft2/irfft2` -> `rfftn/irfftn` (FNO class is unused but keeps the files importable and consistent).
- `train_ddpm/functions/losses.py`: `import torch.fft`; `fft2/irfft2` -> `fftn/irfftn`.

### Command and result

```bash
cd Diffusion-based-Fluid-Super-resolution
python main.py --config kmflow_re1000_rs256_sparse_recons_local.yml --seed 1234 --sample_step 1 --t 240 --r 30 --max_batches 1
```
~33 s total (including loading the 3.35 GB file). Output in `experiments/kmflow_re1000_rs256_local/recons_u3232_nearest_t240_r30_lam0.0/sample_batch0/`.

| | L2 to ground truth | vorticity residual |
|---|---|---|
| input (32x32 nearest) | 2.209 | 2071 |
| SDEdit t=240, 30 DDIM steps (original zero-padded `conv_out`) | 1.904 | 71.2 |
| SDEdit t=240, 30 DDIM steps (`conv_out_padding: circular`, fix 2.9) | **1.900** | **37.8** |
| ground truth | 0 | 0.172 |

Visually the vortex structure and filaments are recovered from the blocky input (`results/recons256_pretrained.png`).

---

## Step 3: downsample to 64x64

```bash
python scripts/downsample_kf.py data/kf_2d_re1000_256_40seed.npy data/kf_2d_re1000_64_40seed.npy --factor 4
```
Output (40, 320, 64, 64) float32, ~10 s.
Checked against a direct 4x4 mean. **The std drops from 4.785 to 4.681** (averaging removes small-scale
vorticity), so the 64 run must use its own normalization statistics (`km64_stats.npz`), not the 256 ones.

Note: `pretrained_weights/km256_stats.npz` contains mean -23.7 / std 22.3, which does not match the data
(mean ~0, std 4.79). The inference runner never reads it (it computes stats from the data). The training copy
`train_ddpm/km256_stats.npz` is correct (std 4.787).

---

## Step 4: training at 64x64

### Config: `train_ddpm/configs/km_re1000_rs64.yml` (new, based on `km_re1000_rs256.yml`, unconditional "simple" model)

| key | 256 original | 64 local | why |
|---|---|---|---|
| `data_dir` | `./data/..._256_...` | `../../data/kf_2d_re1000_64_40seed.npy` | new data |
| `stat_path` | `./km256_stats.npz` | `./km64_stats.npz` | new stats; computed from the 36 training seeds on first run (mean -2.3e-8, std 4.682) |
| `image_size` | 256 | 64 | |
| `ch_mult` | [1,1,1,2] | [1,1,2] | 3 -> 2 downsamplings: 64->32->16 (256 model: 256->128->64->32) |
| `attn_resolutions` | [16] | [] | the 256 model's levels never reach 16, so it had no per-level attention, only mid-block attention; kept the same (mid block is now at 16x16) |
| `ema_rate` | 0.9999 | 0.999 | EMA has no warm-up and starts from the random init; 0.9999^5000 = 0.61 of the init would remain, and inference loads the EMA weights. Use 0.9999 again for a full run |
| `n_epochs` | 300 | 14 | the loop only stops on `n_epochs` (**`n_iters` is never read**); 1 epoch = 36x318/32 = 358 steps, so 14 epochs ≈ 5000 steps |
| `snapshot_freq` | 10000 | 1000 | |
| `num_workers` | 4 | 0 | Windows; the dataset is already in RAM |
| `batch_size` | 32 | 32 | 3.2 GB peak VRAM |

Model size: 3.03 M parameters.

### Problems and fixes

| # | Problem | Cause | Fix |
|---|---|---|---|
| 4.1 | `type: simple` would crash with `TypeError` | `train_ddpm/main.py` hard-codes `ConditionalDiffusion` (the `Diffusion` line is commented out); its train loop calls `loss_registry['simple'](model,x,t,e,b,x_offset,x_scale)`, but `noise_estimation_loss` takes 5 args | `train_ddpm/main.py`: pick runner by `config.model.type` |
| 4.2 | `TypeError: 'dict_keys' object is not subscriptable` at step ~125 | `KMFlowTensorDataset` evicts from its cache with `self.cache.keys()[i]` (Python 2 idiom) once it has >4000 entries | `train_ddpm/datasets/utils.py`: `list(self.cache.keys())[i]` |
| 4.3 | 2.1 s/step (5000 steps ≈ 3 h) | Profiling: `aten::copy_`/`fill_` dominate, not convolutions. Almost every conv uses `padding_mode='circular'` (correct: periodic domain), and PyTorch 1.7's `F._pad_circular` is a slow Python implementation. Zero padding (timing only) gives 0.85 s/step | Bit-identical faster circular pad (one wrap-around `cat` per axis) installed at the top of both model files, guarded by `hasattr(F, '_pad_circular')`: 2.06 -> 1.26 s/step. Model maths and checkpoints are unchanged |
| 4.4 | Remaining slowness | The laptop was on battery; the Max-Q GPU was capped at ~13-28 W | Plug in the charger: **0.33 s/step** at 65 W (5000 steps ≈ 28 min) |
| 4.5 | Unconditional samples noisy, std 2x too large (8.9 vs 4.2), strong border artifacts (border/interior amplitude 3.8) | Inference model's zero-padded `conv_out` vs circular in training (see 2.9); the error accumulates over 100 DDIM steps | `conv_out_padding: circular` -> sample std 4.5, border/interior 0.94 |
| 4.6 | First run died after ~200 steps with `CUDA error: an illegal memory access` | The laptop went to sleep on low battery; on wake-up the GPU driver had been reset and the process's CUDA context was invalid. Not a code bug | Restarted from scratch (no checkpoint before step 1000). Keep the machine from sleeping during training (`powercfg /change standby-timeout-ac 0`); resume with `--resume_training` |

Other observations (not changed):
- The training runner's `sample()` is `pass`; there is no built-in sampling. Hence `scripts/sample_uncond.py`.
- The final save after training (`ckpt_5012.pth`) writes the `states` list from the *last snapshot* (step 5000), not the final weights.
- The repo commits `__pycache__/*.pyc`; they show up as modified after every run.
- `train_ddpm/main.py` with `--ni` on an existing log dir calls `shutil.rmtree(tb_path)` without checking that it exists.

### Command

```bash
cd Diffusion-based-Fluid-Super-resolution/train_ddpm
python -u main.py --config km_re1000_rs64.yml --exp ./experiments/km64_baseline/ --doc km64_baseline --ni > train_km64.log 2>&1
# resume after interruption: add --resume_training
```

### Result

5012 steps (14 epochs), 28 min on AC power. Mean epoch loss (sum of squared eps error over 3x64x64 pixels):

| epoch | 0 | 1 | 2 | 4 | 6 | 8 | 10 | 13 |
|---|---|---|---|---|---|---|---|---|
| loss | 1349 | 665 | 573 | 504 | 491 | 452 | 435 | 417 |

417 / 12288 ≈ 0.034 per-pixel eps-MSE, still decreasing slowly.

Denoising check on test frames (one-step x0 prediction from x_t, EMA weights, circular `conv_out`):

| t | 50 | 100 | 240 | 400 | 700 |
|---|---|---|---|---|---|
| x0 rel. error | 0.066 | 0.107 | 0.206 | 0.346 | 0.855 |
| rel. error of the noisy input itself | 0.198 | 0.385 | 1.020 | 2.291 | 13.5 |

Unconditional DDIM samples (100 steps from pure noise, `scripts/sample_uncond.py` ->
`results/samples_uncond_ema.png`) look like Kolmogorov-flow vortices (cores, filaments, shear layers).
A remaining sign of under-training: some samples have a non-zero mean vorticity (whole field shifted positive or
negative), while the real data has zero mean in every frame (periodic domain). This also explains sample std 5.7 vs 4.3.

```bash
python scripts/sample_uncond.py --lib Diffusion-based-Fluid-Super-resolution \
  --config configs/kmflow_re1000_rs64_sparse_recons_local.yml \
  --ckpt train_ddpm/experiments/km64_baseline/logs/km64_baseline/ckpt.pth \
  --data data/kf_2d_re1000_64_40seed.npy --n 8 --steps 100 --out results/samples_uncond_ema.png
```

---

## Step 5: 64 -> 16 -> 64 reconstruction test

```bash
python scripts/make_lowres_input.py data/kf_2d_re1000_64_40seed.npy data/kmflow_lowres_u16_from64.npz --factor 4 --down avg
cd Diffusion-based-Fluid-Super-resolution
python main.py --config kmflow_re1000_rs64_sparse_recons_local.yml --t 240 --r 30 --max_batches 5
python ../scripts/compare_recons.py experiments/kmflow_re1000_rs64_local/recons_u1616_nearest_t240_r30_lam0.0 --factor 4
```

- Low-res: 4x4 average pooling (consistent with step 3), nearest-upsampled to 64 as the SDEdit input (same format as the paper's input).
- `configs/kmflow_re1000_rs64_sparse_recons_local.yml` (new): 64 data/input paths, `data_kw: u1616_nearest`, model section identical to the training config, `ckpt_path` -> our checkpoint, `batch_size: 20`.
- Bicubic baseline: periodic (circular pad of 2 LR cells -> bicubic x4 -> crop), computed from the same 16x16 field.

### Result

100 test frames (first 5 batches of 20, all from the first test seed, index 36), 30 DDIM reverse steps, `sample_step 1`, no physics guidance.
Figures and `metrics.txt`: `results/recons64_t{160,240,320,400}/`.

| method | RMSE | rel. L2 | runner vorticity residual |
|---|---|---|---|
| nearest (= SDEdit input) | 2.943 ± 0.275 | 0.538 ± 0.041 | |
| bicubic (periodic) | **2.084 ± 0.464** | **0.380 ± 0.082** | |
| SDEdit t=160 | 2.633 ± 0.360 | 0.481 ± 0.059 | 102.1 |
| SDEdit t=240 | 2.559 ± 0.399 | 0.467 ± 0.067 | 60.4 |
| SDEdit t=320 | 2.550 ± 0.434 | 0.465 ± 0.074 | 41.5 |
| SDEdit t=400 | 2.618 ± 0.472 | 0.478 ± 0.081 | 31.7 |

Interpretation:
- **Pointwise, SDEdit beats its own input (nearest) but not bicubic.** The best t is about 240-320. Larger t gives a lower PDE residual (more "physical" fields) but drifts further from the measurement.
- **Spectrally, SDEdit is much better** (`energy_spectrum.png`): above the LR Nyquist wavenumber (k > 8) bicubic's energy falls by 2-6 orders of magnitude, while SDEdit follows the ground-truth spectrum closely. SDEdit generates realistic small-scale vortices and filaments, but 16x16 does not pin down *where* they are, and a pointwise metric penalizes small structures in slightly wrong places twice (once where they are missing, once where they were put). Bicubic is smooth, so it has the lowest MSE.
- Faint straight block edges from the nearest-upsampled input are still visible in some SDEdit outputs.
- Caveats vs. the paper: 4x from 16x16 (about 4 px per forcing wavelength) is a much less informed problem than the paper's 8x from 32x32 at 256; the model only had 5k steps with no physics conditioning; and `sample_step 1` (the paper also repeats the SDEdit loop).
