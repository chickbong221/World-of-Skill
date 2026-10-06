import gymnasium as gym

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common


class NonPrivilegedObsWrapper(gym.ObservationWrapper):
    """Remove simulator-only privileged fields from obs['extra']."""

    PRIVILEGED_KEYS = {
        # PickCube-style keys
        'is_grasped', 'goal_pos', 'obj_pose',
        'tcp_to_obj_pos', 'obj_to_goal_pos',
        # MSHAB-style keys
        'obj_pose_wrt_base', 'goal_pos_wrt_base',
    }

    def __init__(self, env) -> None:
        super().__init__(env)
        self._base_env: BaseEnv = env.unwrapped
        init_raw_obs = common.to_tensor(self._base_env._init_raw_obs)
        self._base_env.update_obs_space(self.observation(init_raw_obs))

    def observation(self, obs):
        if 'extra' in obs:
            obs = dict(obs)
            obs['extra'] = {k: v for k, v in obs['extra'].items()
                            if k not in self.PRIVILEGED_KEYS}
        return obs
