import contextlib
import io
import logging
import warnings

import elements
import embodied
import numpy as np


_ENV_NOISE_SILENCED = False


def _silence_env_noise():
  global _ENV_NOISE_SILENCED
  if _ENV_NOISE_SILENCED:
    return
  _ENV_NOISE_SILENCED = True
  warnings.filterwarnings("ignore", message=".*Box bound precision.*")
  warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
  for name in ("robosuite", "robosuite.utils", "robosuite.macros"):
    logging.getLogger(name).setLevel(logging.ERROR)


class OfflineComp(embodied.Env):

  def __init__(self, task, obs_dim=93, action_dim=8):
    del task
    self.obs_dim = int(obs_dim)
    self.action_dim = int(action_dim)

  @property
  def obs_space(self):
    return {
        "vector": elements.Space(np.float32, (self.obs_dim,)),
        "reward": elements.Space(np.float32),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
        "is_terminal": elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        "action": elements.Space(np.float32, (self.action_dim,), -1.0, 1.0),
        "reset": elements.Space(bool),
    }

  def step(self, action):
    raise RuntimeError("OfflineComp only defines spaces; use script=train_offline")


class CompoSuiteEval(embodied.Env):
  """Live MuJoCo CompoSuite env for periodic offline-eval rollouts.

  Wraps composuite.make(robot, obj, obstacle, task, ...) and exposes the
  embodied step interface: vector obs, reward, is_first/is_last/is_terminal,
  plus a log/success scalar sourced from the underlying robosuite
  `_check_success()` predicate.
  """

  def __init__(
      self, robot, obj, obstacle, objective,
      obs_dim=93, action_dim=8, horizon=500):
    _silence_env_noise()
    with contextlib.redirect_stdout(io.StringIO()), \
        contextlib.redirect_stderr(io.StringIO()), \
        warnings.catch_warnings():
      warnings.simplefilter("ignore")
      import composuite  # noqa: F401  (imports and registers subtasks)
      self._robot = robot
      self._obj = obj
      self._obstacle = None if obstacle == "None" else obstacle
      self._objective = objective
      self.obs_dim = int(obs_dim)
      self.action_dim = int(action_dim)
      self._horizon = int(horizon)
      self._env = composuite.make(
          robot=self._robot,
          obj=self._obj,
          obstacle=self._obstacle,
          task=self._objective,
          controller="joint",
          env_horizon=self._horizon,
          has_renderer=False,
          has_offscreen_renderer=False,
          reward_shaping=True,
          ignore_done=True,
          use_camera_obs=False,
          use_task_id_obs=True,
      )
    self._done = True
    self._steps = 0
    obs_shape = self._env.observation_space.shape
    if obs_shape != (self.obs_dim,):
      raise ValueError(
          f"CompoSuite obs dim {obs_shape} != expected ({self.obs_dim},) "
          f"for task {self.task_name}")
    act_shape = self._env.action_space.shape
    if act_shape != (self.action_dim,):
      raise ValueError(
          f"CompoSuite action dim {act_shape} != expected ({self.action_dim},) "
          f"for task {self.task_name}")

  @property
  def task_name(self):
    obstacle = self._obstacle if self._obstacle is not None else "None"
    return f"{self._robot}_{self._obj}_{obstacle}_{self._objective}"

  @property
  def obs_space(self):
    return {
        "vector": elements.Space(np.float32, (self.obs_dim,)),
        "reward": elements.Space(np.float32),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
        "is_terminal": elements.Space(bool),
        "log/success": elements.Space(np.float32),
    }

  @property
  def act_space(self):
    return {
        "action": elements.Space(np.float32, (self.action_dim,), -1.0, 1.0),
        "reset": elements.Space(bool),
    }

  def step(self, action):
    if action["reset"] or self._done:
      obs = self._env.reset()
      self._done = False
      self._steps = 0
      return self._obs(obs, 0.0, is_first=True)
    act = np.clip(
        np.asarray(action["action"], np.float32), -1.0, 1.0)
    obs, reward, done, _ = self._env.step(act)
    self._steps += 1
    truncated = self._steps >= self._horizon
    self._done = bool(done) or truncated
    is_terminal = bool(done) and not truncated
    return self._obs(
        obs, reward, is_last=self._done, is_terminal=is_terminal)

  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    try:
      success = bool(self._env.env._check_success())
    except Exception:
      success = False
    return {
        "vector": np.asarray(obs, np.float32),
        "reward": np.float32(reward),
        "is_first": bool(is_first),
        "is_last": bool(is_last),
        "is_terminal": bool(is_terminal),
        "log/success": np.float32(success),
    }

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass
