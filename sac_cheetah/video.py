from pathlib import Path

import imageio.v2 as imageio

from sac_cheetah.envs import make_env


def record_video(model, env_id: str, path: str, seed: int, steps: int) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(env_id, seed, render_mode="rgb_array")
    fps = env.metadata.get("render_fps", 30)

    try:
        obs, _ = env.reset(seed=seed)
        with imageio.get_writer(out, fps=fps) as video:
            video.append_data(env.render())

            for _ in range(steps):
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, _, terminated, truncated, _ = env.step(action)
                video.append_data(env.render())

                if terminated or truncated:
                    break
    finally:
        env.close()
