import time

import elements
import embodied

from offline_comp.dataset import make_datasets

from . import env_eval


class Timed:
  """Stream wrapper that accumulates the time spent waiting for batches."""

  def __init__(self, stream):
    self.stream = stream
    self.wait = 0.0

  def __iter__(self):
    return self

  def __next__(self):
    start = time.perf_counter()
    batch = next(self.stream)
    self.wait += time.perf_counter() - start
    return batch


class StepClock:
  """Step-based analogue of embodied.LocalClock: fires once step advances by
  at least `every` since the previous fire. `every <= 0` disables the clock."""

  def __init__(self, every, first=False, start=None):
    self.every = int(every)
    self.first = first
    self.prev = start

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
    per_update = args.batch_size * args.batch_length
    unit = getattr(args, 'step_unit', 'samples')  # 'updates': step counts updates
    batch_steps = 1 if unit == 'updates' else per_update
    if getattr(args, 'clock', 'time') == 'step':
      Clock = lambda every: StepClock(
          every, start=0 if unit == 'updates' else None)
    else:
      Clock = embodied.LocalClock
    # two_phase: world model for wm_steps updates, then the policy.
    phased = getattr(args, 'schedule', 'joint') == 'two_phase'
    wm_steps = int(args.wm_steps) if phased else 0
    assert not phased or unit == 'updates', 'two_phase counts updates'
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
    train_stream = Timed(iter(agent.stream(train_stream)))
    perf = [time.time(), 0.0, 0]  # time, data wait and step at the last log
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
      phase = ('wm' if int(step) < wm_steps else 'policy') if phased else None
      if phased and int(step) == wm_steps:
        print(f"[phase] world model trained for {wm_steps} updates, "
              f"training the policy on it for {int(args.steps) - wm_steps}")
      batch = next(train_stream)
      carry_train, outs, mets = agent.train(
          carry_train, batch, **({'phase': phase} if phased else {}))
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

      # Env evals follow the policy updates only (all updates when joint).
      pstep = int(step) - wm_steps
      if pstep > 0 and should_env_eval(pstep):
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
            ev = env_eval.evaluate(
                agent, {'train': train_tasks, 'test': test_tasks},
                episodes, horizon, max_tasks,
                slots=getattr(args, 'env_eval_slots', 1),
                workers=getattr(args, 'env_eval_workers', 0))
            logger.add(ev['train'], prefix='env_eval/train')
            logger.add(ev['test'], prefix='env_eval/test')
            print(
                f"[env_eval] done in {time.time() - eval_start:.1f}s | "
                f"train success_once={ev['train'].get('success_once', 0.0):.2f} "
                f"test success_once={ev['test'].get('success_once', 0.0):.2f}")
          except Exception as exc:
            import traceback
            print(f"[env_eval] FAILED after {time.time() - eval_start:.1f}s: "
                  f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

      if should_log(step):
        now, wait = time.time(), train_stream.wait
        span = max(now - perf[0], 1e-6)
        logger.add({
            "steps_per_sec": (int(step) - perf[2]) / span,
            "data_wait_frac": (wait - perf[1]) / span,
        }, prefix="perf")
        perf[:] = [now, wait, int(step)]
        updates = int(step) if unit == 'updates' else int(step) // per_update
        count = {"updates": updates, "transitions": updates * per_update}
        if phased:
          count["policy_updates"] = max(updates - wm_steps, 0)
          count["phase_policy"] = float(updates > wm_steps)
        logger.add(count, prefix="offline")
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