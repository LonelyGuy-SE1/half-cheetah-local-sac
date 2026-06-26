
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    env_id: str = "HalfCheetah-v5"
    seed: int = 7
    total_steps: int = 300_000
    video_steps: int = 1000
    model_path: str = "models/sac_half_cheetah"
    initial_video: str = "videos/initial.mp4"
    final_video: str = "videos/final.mp4"
