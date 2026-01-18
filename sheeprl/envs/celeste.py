import gymnasium as gym
import celeste_ai_gym
from celeste_ai_gym import CelesteEnv
from celeste_ai_gym.action_wrapper import CelesteActionWrapper, DASHLESS_SIMPLE, DASHLESS_COMPLEX, DASH_RESTRICTED

ACTIONS_SPACE_MAP = {
    "dashless_simple": DASHLESS_SIMPLE,
    "dashless_complex": DASHLESS_COMPLEX,
    "dash_restricted": DASH_RESTRICTED
}

class CelesteWrapper(gym.Wrapper):
    def __init__(self, action_space: str = "dashless_simple", render_mode: str = "rgb_array"):
        env = CelesteEnv(render_mode=render_mode)
        env = CelesteActionWrapper(env, ACTIONS_SPACE_MAP[action_space])
        super().__init__(env)

        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    env.observation_space.low,
                    env.observation_space.high,
                    env.observation_space.shape,
                    env.observation_space.dtype,
                )
            }
        )

    def step(self, action: int) -> tuple[any, float, bool, bool, dict]:
        obs, reward, done, truncated, info = self.env.step(action)
        converted_obs = {"rgb": obs.copy()}
        return converted_obs, reward, done, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[any, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        converted_obs = {"rgb": obs.copy()}
        return converted_obs, info