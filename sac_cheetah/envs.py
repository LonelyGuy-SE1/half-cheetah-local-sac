import os
from math import pi

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor


def wrap_pi(x: float) -> float:
    return ((x + pi) % (2 * pi)) - pi


class AntiFlipWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        data = self.unwrapped.data

        torso_z = float(data.xpos[1, 2])
        rootz = wrap_pi(float(data.qpos[1]))
        rooty = wrap_pi(float(data.qpos[2]))

        angle_cost = 0.5 * rootz**2 + 2.0 * rooty**2
        height_cost = 4.0 * max(0.0, 0.45 - torso_z) ** 2
        fell = torso_z < 0.30 or abs(rooty) > 1.20

        reward = float(reward) - angle_cost - height_cost
        if fell:
            reward -= 25.0
            terminated = True

        info["torso_z"] = torso_z
        info["rootz"] = rootz
        info["rooty"] = rooty
        info["anti_flip_cost"] = angle_cost + height_cost
        info["fell"] = fell
        return obs, reward, terminated, truncated, info


def make_env(env_id: str, seed: int, render_mode: str | None = None):
    env = AntiFlipWrapper(gym.make(env_id, render_mode=render_mode))
    env.action_space.seed(seed)
    return env


def make_train_env(env_id: str, seed: int):
    return Monitor(make_env(env_id, seed))
