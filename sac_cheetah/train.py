import argparse
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed

from sac_cheetah.config import TrainConfig
from sac_cheetah.envs import make_train_env
from sac_cheetah.video import record_video


def pick_device(requested: str) -> str:
    cuda_ok = cuda_ready()
    if requested == "cuda" and not cuda_ok:
        raise RuntimeError("CUDA was requested, but a CUDA tensor op failed.")
    if requested != "auto":
        return requested
    return "cuda" if cuda_ok else "cpu"


def cuda_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.ones(1, device="cuda")
        return float((x + 1).cpu()[0]) == 2.0
    except Exception as exc:
        print(f"cuda check failed: {exc}")
        return False


def parse_args() -> argparse.Namespace:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default=cfg.env_id)
    parser.add_argument("--steps", type=int, default=cfg.total_steps)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--video-steps", type=int, default=cfg.video_steps)
    parser.add_argument("--model-path", default=cfg.model_path)
    parser.add_argument("--initial-video", default=cfg.initial_video)
    parser.add_argument("--final-video", default=cfg.final_video)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        env_id=args.env_id,
        seed=args.seed,
        total_steps=args.steps,
        video_steps=args.video_steps,
        model_path=args.model_path,
        initial_video=args.initial_video,
        final_video=args.final_video,
    )
    device = pick_device(args.device)

    if device == "cuda":
        print(f"device: cuda, {torch.cuda.get_device_name(0)}")
    else:
        print("device: cpu")

    set_random_seed(cfg.seed)
    Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"recording {cfg.initial_video}")
    record_video(None, cfg.env_id, cfg.initial_video, cfg.seed, cfg.video_steps)

    env = make_train_env(cfg.env_id, cfg.seed)
    model = SAC("MlpPolicy", env, seed=cfg.seed, device=device, verbose=1)
    model.learn(total_timesteps=cfg.total_steps)
    model.save(cfg.model_path)
    env.close()

    print(f"saved {cfg.model_path}.zip")
    print(f"recording {cfg.final_video}")
    record_video(model, cfg.env_id, cfg.final_video, cfg.seed + 1, cfg.video_steps)


if __name__ == "__main__":
    main()
