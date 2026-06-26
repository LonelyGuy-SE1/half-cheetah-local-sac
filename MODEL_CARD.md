---
library_name: stable-baselines3
pipeline_tag: reinforcement-learning
tags:
  - reinforcement-learning
  - stable-baselines3
  - sac
  - gymnasium
  - mujoco
  - halfcheetah
  - continuous-control
---

# SAC HalfCheetah-v5

Stable Baselines3 SAC policy trained on Gymnasium `HalfCheetah-v5`.

The environment includes a small anti-flip reward guard to discourage belly-slide exploit postures.

## Videos

### Before training

<video controls src="https://huggingface.co/Lonelyguyse1/half-cheetah-local-sac-test/resolve/main/videos/initial.mp4"></video>

### After training

<video controls src="https://huggingface.co/Lonelyguyse1/half-cheetah-local-sac-test/resolve/main/videos/final.mp4"></video>

## What Was Done

- Trained SAC with `MlpPolicy` for `300000` timesteps.
- Recorded a random policy rollout before training.
- Recorded the trained policy rollout after training.
- Saved the trained checkpoint as `sac_half_cheetah.zip`.

## Evaluation

Single deterministic rollout with seed `8`.

| Metric | Value |
| --- | ---: |
| Steps | 1000 |
| Return | 7031.927 |
| Mean reward | 7.032 |
| Mean x velocity | 7.465 |
| Final x position | 373.190 |
| Minimum torso height | 0.534 |
| Maximum absolute root angle | 0.269 |
| Fell | false |

## Load

```python
from stable_baselines3 import SAC

model = SAC.load("sac_half_cheetah.zip")
```

Use the included wrapper for matching evaluation.

```python
from sac_cheetah.config import TrainConfig
from sac_cheetah.envs import make_env

cfg = TrainConfig()
env = make_env(cfg.env_id, cfg.seed + 1, render_mode="rgb_array")
```
