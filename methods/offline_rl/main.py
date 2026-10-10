import os
import pathlib
import sys
from functools import partial as bind

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import portal
import ruamel.yaml as yaml


def main(argv=None):
  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=os.path.expanduser(
      config.logdir.format(timestamp=elements.timestamp())))
  print('Algorithm:', config.agent.algo)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  logdir.mkdir()
  config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      seed=config.seed,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      replay_context=config.replay_context,
      data=config.get('data', {}),
  )
  assert config.script == 'train_offline', config.script
  embodied.run.train_offline(
      bind(make_agent, config), bind(make_logger, config), args)


def make_agent(config, obs_space, act_space):
  from .agent import Agent
  agent = config.agent
  if agent.anneal < 0:  # one cosine decay over the whole run, in actor steps
    updates = int(config.run.steps)
    if config.run.step_unit != 'updates':
      updates //= config.batch_size * config.batch_length
    if agent.algo == 'td3bc':
      updates //= agent.policy_freq
    agent = agent.update(anneal=updates)
  return Agent(obs_space, act_space, elements.Config(
      **agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def make_logger(config):
  step = elements.Counter()
  logdir, lc = config.logdir, config.logger
  outputs = [elements.logger.TerminalOutput(lc.filter, 'Agent')]
  for output in lc.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
    elif output == 'wandb':
      import wandb
      run_name = lc.wandb_name or '/'.join(logdir.split('/')[-4:])
      os.environ.update(WANDB_DIR=logdir, WANDB_RESUME='allow')
      wandb.init(
          project=lc.wandb_project or 'offline-comp-baselines',
          entity=lc.wandb_entity or None, name=run_name,
          group=lc.wandb_group or None, dir=logdir, config=dict(config),
          resume='allow')
      outputs.append(elements.logger.WandBOutput(
          '/'.join(logdir.split('/')[-4:])))
    else:
      raise NotImplementedError(output)
  return elements.Logger(step, outputs, 1)


if __name__ == '__main__':
  main()
