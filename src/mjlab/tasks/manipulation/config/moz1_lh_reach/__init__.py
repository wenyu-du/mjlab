from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import moz1_lh_reach_env_cfg
from .rl_cfg import moz1_lh_reach_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Moz1-Lh-Reach",
  env_cfg=moz1_lh_reach_env_cfg(),
  play_env_cfg=moz1_lh_reach_env_cfg(play=True),
  rl_cfg=moz1_lh_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
