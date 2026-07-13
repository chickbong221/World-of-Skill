import time

import elements
import embodied
import numpy as np

from offline_comp.dataset import make_datasets


def _run_env_eval(agent, tasks, episodes, horizon, max_tasks):
  """Roll out `agent.policy` in live CompoSuite envs and return metrics.

  Runs `episodes` episodes per task (capped at `max_tasks` tasks) and
  aggregates per-task and mean success_once / success_at_end / return.
  """
  from embodied.envs.offline_comp import CompoSuiteEval

  if max_tasks and max_tasks > 0:
    tasks = list(tasks)[:max_tasks]
  else:
    tasks = list(tasks)

  per_task = {}
  agg_ret, agg_once, agg_end = [], [], []
  for task in tasks:
    env = CompoSuiteEval(
        task.robot, task.obj, task.obstacle, task.objective,
        horizon=horizon)
    try:
      returns, once, ends = _rollout(agent, env, episodes, task)
    finally:
      env.close()
    per_task[task.name] = {
        'return': float(np.mean(returns)) if returns else 0.0,
        'success_once': float(np.mean(once)) if once else 0.0,
        'success_at_end': float(np.mean(ends)) if ends else 0.0,
    }
    agg_ret.extend(returns)
    agg_once.extend(once)
    agg_end.extend(ends)

  metrics = {}
  if agg_ret:
    metrics['return'] = float(np.mean(agg_ret))
    metrics['success_once'] = float(np.mean(agg_once))
    metrics['success_at_end'] = float(np.mean(agg_end))
    metrics['tasks_evaluated'] = float(len(tasks))
  for name, values in per_task.items():
    for key, value in values.items():
      metrics[f'tasks/{name}/{key}'] = value
  return metrics


def _rollout(agent, env, episodes, task=None):
  """Run `episodes` episodes in `env` under `agent.policy(mode='eval')`."""
  carry = agent.init_policy(1)
  action_shape = env.act_space['action'].shape
  reset_action = {
      'action': np.zeros(action_shape, np.float32),
      'reset': True,
  }
  obs = env.step(reset_action)
  # MoSS: agent.policy() asserts obs.keys() == obs_space.keys(), and
  # obs_space now carries task ids. The policy does not consume them (they
  # are excluded from the encoder); they only need to be present and finite.
  extra = {}
  if getattr(agent, 'obs_space', None) and 'task_id' in agent.obs_space:
    from offline_comp import tasks as tasks_mod
    tid = 0
    axes = np.zeros((4,), np.int32)
    if task is not None:
      axes = np.array([
          tasks_mod.ROBOTS.index(task.robot),
          tasks_mod.OBJECTS.index(task.obj),
          tasks_mod.OBSTACLES.index(task.obstacle),
          tasks_mod.OBJECTIVES.index(task.objective),
      ], np.int32)
    extra = {
        'task_id': np.asarray(tid, np.int32)[None],
        'task_axes': axes[None],
    }

  returns, once, ends = [], [], []
  ep_return = 0.0
  ep_success_once = False
  ep_last_success = 0.0
  while len(returns) < episodes:
    log_success = float(obs.get('log/success', 0.0))
    ep_last_success = log_success
    if log_success > 0.5:
      ep_success_once = True
    policy_obs = {
        k: np.asarray(v)[None] for k, v in obs.items()
        if not k.startswith('log/')}
    policy_obs.update(extra)
    carry, acts, _ = agent.policy(carry, policy_obs, mode='eval')
    act = {k: np.asarray(v)[0] for k, v in acts.items()}
    act['reset'] = False
    obs = env.step(act)
    ep_return += float(obs['reward'])
    if bool(obs['is_last']):
      log_success = float(obs.get('log/success', 0.0))
      ep_last_success = log_success
      if log_success > 0.5:
        ep_success_once = True
      returns.append(ep_return)
      once.append(1.0 if ep_success_once else 0.0)
      ends.append(ep_last_success)
      ep_return = 0.0
      ep_success_once = False
      ep_last_success = 0.0
      if len(returns) < episodes:
        obs = env.step(reset_action)
  return returns, once, ends


class StepClock:
  """Step-based analogue of embodied.LocalClock: fires once step advances by
  at least `every` since the previous fire. `every <= 0` disables the clock."""

  def __init__(self, every, first=False):
    self.every = int(every)
    self.first = first
    self.prev = None

  def __call__(self, step):
    if self.every <= 0:
      return False
    step = int(step)
    if self.prev is None:
      self.prev = step
      return self.first
    if step >= self.prev + self.every:
      self.prev = step
      return True
    return False


def train_offline(make_agent, make_logger, args):
  train_data, train_report_data, test_report_data, train_tasks, test_tasks = (
      make_datasets(
          args.data,
          args.batch_length + args.replay_context,
          args.report_length + args.replay_context,
          seed=args.seed, logdir=args.logdir))
  try:
    agent = make_agent(train_data.obs_space, train_data.act_space)
    logger = make_logger()
    step = logger.step
    usage = elements.Usage(**args.usage)
    train_agg = elements.Agg()
    batch_steps = args.batch_size * args.batch_length
    clock_mode = getattr(args, 'clock', 'time')
    Clock = StepClock if clock_mode == 'step' else embodied.LocalClock
    should_log = Clock(args.log_every)
    should_report = Clock(args.report_every)
    should_eval = Clock(args.eval_every)
    should_env_eval = Clock(getattr(args, 'env_eval_every', 0))

    train_stream = embodied.streams.Stateless(
        lambda: train_data.sample(args.batch_size))
    train_report_stream = embodied.streams.Stateless(
        lambda: train_report_data.sample(args.batch_size))
    test_report_stream = embodied.streams.Stateless(
        lambda: test_report_data.sample(args.batch_size))
    train_stream = iter(agent.stream(train_stream))
    train_report_stream = iter(agent.stream(train_report_stream))
    test_report_stream = iter(agent.stream(test_report_stream))
    carry_train = agent.init_train(args.batch_size)
    carry_report = agent.init_report(args.batch_size)

    cp = elements.Checkpoint(elements.Path(args.logdir) / "ckpt")
    cp.step = step
    cp.agent = agent
    if args.from_checkpoint:
      elements.checkpoint.load(args.from_checkpoint, dict(
          agent=agent.load))
    if (elements.Path(args.logdir) / "ckpt").exists():
      cp.load()

    print("Offline CompoSuite train tasks:", len(train_tasks))
    print("Offline CompoSuite test tasks:", len(test_tasks))
    print("Start offline Dreamer training loop")
    start = time.time()

    while step < args.steps:
      batch = next(train_stream)
      carry_train, outs, mets = agent.train(carry_train, batch)
      if "replay" in outs:
        pass
      train_agg.add(mets, prefix="train")
      step.increment(batch_steps)

      if should_eval(step):
        carry_report, mets = agent.report(
            carry_report, next(test_report_stream))
        logger.add(mets, prefix="eval")
        logger.add({
            "tasks/train": len(train_tasks),
            "tasks/test": len(test_tasks),
            "total_time": time.time() - start,
        }, prefix="offline")

      if should_report(step):
        carry_report, mets = agent.report(
            carry_report, next(train_report_stream))
        logger.add(mets, prefix="report")

      if should_env_eval(step):
        episodes = int(getattr(args, 'env_eval_episodes', 0) or 0)
        horizon = int(getattr(args, 'env_eval_horizon', 500) or 500)
        max_tasks = int(getattr(args, 'env_eval_max_tasks', 0) or 0)
        if episodes > 0:
          eval_start = time.time()
          print(
              f"[env_eval @ step {int(step)}] rolling out "
              f"{episodes} eps x <= {max_tasks or len(train_tasks)} train "
              f"and <= {max_tasks or len(test_tasks)} test tasks")
          try:
            train_mets = _run_env_eval(
                agent, train_tasks, episodes, horizon, max_tasks)
            test_mets = _run_env_eval(
                agent, test_tasks, episodes, horizon, max_tasks)
            logger.add(train_mets, prefix='env_eval/train')
            logger.add(test_mets, prefix='env_eval/test')
            print(
                f"[env_eval] done in {time.time() - eval_start:.1f}s | "
                f"train success_once={train_mets.get('success_once', 0.0):.2f} "
                f"test success_once={test_mets.get('success_once', 0.0):.2f}")
          except Exception as exc:
            import traceback
            print(f"[env_eval] FAILED after {time.time() - eval_start:.1f}s: "
                  f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

      if should_log(step):
        logger.add(train_agg.result())
        logger.add(train_data.stats(), prefix="dataset/train")
        logger.add(test_report_data.stats(), prefix="dataset/test")
        logger.add(usage.stats(), prefix="usage")
        logger.add({"timer": elements.timer.stats()["summary"]})
        logger.write()

    logger.close()
  finally:
    train_data.close()
    train_report_data.close()
    test_report_data.close()