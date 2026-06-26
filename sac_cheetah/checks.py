import importlib.metadata as md

import gymnasium as gym
import torch


def version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "missing"


def main() -> None:
    print(f"torch: {version('torch')}")
    print(f"gymnasium: {version('gymnasium')}")
    print(f"mujoco: {version('mujoco')}")
    print(f"stable-baselines3: {version('stable-baselines3')}")
    print(f"cuda: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        x = torch.ones(1024, device="cuda")
        print(f"cuda_kernel: {float((x * 2).sum().cpu()) == 2048.0}")

    env = gym.make("HalfCheetah-v5")
    obs, _ = env.reset(seed=7)
    action = env.action_space.sample()
    env.step(action)
    env.close()
    print(f"env: ok, obs_shape={obs.shape}")
