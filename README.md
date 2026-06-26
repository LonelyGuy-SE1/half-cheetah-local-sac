# half-cheetah-local-sac-test

Minimal SAC training for Gymnasium `HalfCheetah-v5` with Stable Baselines3.

PyTorch is pinned to `2.7.1+cu118` so the Quadro P520 can run CUDA kernels.
The env adds a small anti-flip reward guard to stop belly-slide exploits.

## Setup

```bash
uv run --python 3.11 cheetah-checks
```

## Train

```bash
uv run --python 3.11 cheetah-train
```

Outputs:

- `videos/initial.mp4`
- `videos/final.mp4`
- `models/sac_half_cheetah.zip`

Use fewer steps for a quick smoke test:

```bash
uv run --python 3.11 cheetah-train --steps 1000 --model-path /tmp/cheetah_smoke --initial-video /tmp/initial.mp4 --final-video /tmp/final.mp4
```
