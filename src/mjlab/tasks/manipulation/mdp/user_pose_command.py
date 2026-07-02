from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  sample_uniform,
  quat_mul,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UserPoseCommand(CommandTerm):
  cfg: UserPoseCommandCfg

  def __init__(self, cfg: UserPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.asset_name]
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_quat = torch.zeros(self.num_envs, 4, device=self.device)
    self.target_quat[:, 0] = 1.0
    
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    target_body_ids, _ = self.robot.find_bodies("target")
    if len(target_body_ids) > 0:
      global_body_id = int(self.robot.indexing.body_ids[target_body_ids[0]].item())
      self.target_mocap_id = int(self._env.sim.mj_model.body_mocapid[global_body_id])
    else:
      self.target_mocap_id = -1

  @property
  def command(self) -> torch.Tensor:
    local_target_pos = self.target_pos - self._env.scene.env_origins
    return torch.cat([local_target_pos, self.target_quat], dim=-1)

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    r = self.cfg.ranges
    lower_pos = torch.tensor([r.pos_x[0], r.pos_y[0], r.pos_z[0]], device=self.device)
    upper_pos = torch.tensor([r.pos_x[1], r.pos_y[1], r.pos_z[1]], device=self.device)
    target_pos = sample_uniform(lower_pos, upper_pos, (n, 3), device=self.device)
    
    roll = sample_uniform(r.roll[0], r.roll[1], (n,), device=self.device)
    pitch = sample_uniform(r.pitch[0], r.pitch[1], (n,), device=self.device)
    yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
    target_quat = quat_from_euler_xyz(roll, pitch, yaw)

    self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]
    self.target_quat[env_ids] = target_quat

    if self.target_mocap_id != -1:
      self.robot.data.data.mocap_pos[env_ids, self.target_mocap_id] = self.target_pos[env_ids]
      self.robot.data.data.mocap_quat[env_ids, self.target_mocap_id] = self.target_quat[env_ids]

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    for batch in env_indices:
      target_pos = self.target_pos[batch].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.03,
        color=(0.0, 1.0, 0.0, 0.5),
        label=f"target_position_{batch}",
      )

@dataclass(kw_only=True)
class UserPoseCommandCfg(CommandTermCfg):
  asset_name: str = "robot"
  body_name: str = "ee_link"
  
  @dataclass
  class Ranges:
    pos_x: tuple[float, float] = (0.2, 0.6)
    pos_y: tuple[float, float] = (-0.5, 0.5)
    pos_z: tuple[float, float] = (0.3, 0.8)
    roll: tuple[float, float] = (-1.57, 1.57)
    pitch: tuple[float, float] = (-1.57, 1.57)
    yaw: tuple[float, float] = (-1.57, 1.57)

  ranges: Ranges = field(default_factory=Ranges)

  def build(self, env: ManagerBasedRlEnv) -> UserPoseCommand:
    return UserPoseCommand(self, env)
