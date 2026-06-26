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

This is a Stable Baselines3 SAC policy trained locally on Gymnasium `HalfCheetah-v5`.

The environment uses a small anti-flip reward guard. It penalizes extreme torso plane angle and low torso height, then terminates clear fall or belly-slide exploit postures.

## Files

- `sac_half_cheetah.zip`: Stable Baselines3 SAC checkpoint.
- `videos/initial.mp4`: random policy before training.
- `videos/final.mp4`: trained policy rollout.
- `sac_cheetah/`: minimal training, environment, check, and video code.
- `pyproject.toml`: Python dependency setup.

## Training

- Algorithm: Soft Actor-Critic
- Implementation: Stable Baselines3
- Policy: `MlpPolicy`
- Environment: `HalfCheetah-v5`
- Timesteps: `300000`
- Seed: `7`
- Device: Quadro P520 with `torch==2.7.1+cu118`

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

These numbers are a local smoke evaluation, not a benchmark sweep.

## Load

```python
from stable_baselines3 import SAC

model = SAC.load("sac_half_cheetah.zip", device="auto")
```

Use the wrapper in `sac_cheetah.envs` if you want evaluation to match this model card.

```python
from sac_cheetah.config import TrainConfig
from sac_cheetah.envs import make_env

cfg = TrainConfig()
env = make_env(cfg.env_id, cfg.seed + 1, render_mode="rgb_array")
```

## Limitations

This policy is only tested on Gymnasium `HalfCheetah-v5` with the included anti-flip wrapper. It is not meant for real robots, safety-critical systems, or transfer to other MuJoCo tasks without retraining.
