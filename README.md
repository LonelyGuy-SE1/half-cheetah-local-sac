# SAC HalfCheetah-v5

Stable Baselines3 SAC policy trained on Gymnasium `HalfCheetah-v5`.

The environment includes a small anti-flip reward guard to discourage belly-slide exploit postures.

## What Was Done

- Trained SAC with `MlpPolicy` for `300000` timesteps.
- Recorded a random policy rollout before training.
- Recorded the trained policy rollout after training.
- Saved the trained checkpoint as `models/sac_half_cheetah.zip`.

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

## Run

```bash
uv run --python 3.11 cheetah-train
```
