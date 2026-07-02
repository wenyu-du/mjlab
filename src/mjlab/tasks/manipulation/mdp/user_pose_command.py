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

    ee_body_ids, _ = self.robot.find_bodies(cfg.body_name)
    self.ee_body_id = ee_body_ids[0] if len(ee_body_ids) > 0 else -1

    target_body_ids, _ = self.robot.find_bodies("target")
    if len(target_body_ids) > 0:
      global_body_id = int(self.robot.indexing.body_ids[target_body_ids[0]].item())
      self.target_mocap_id = int(self._env.sim.mj_model.body_mocapid[global_body_id])
    else:
      self.target_mocap_id = -1

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.target_pos, self.target_quat], dim=-1)

  def _update_metrics(self) -> None:
    if self.ee_body_id != -1:
      ee_pos = self.robot.data.body_link_pos_w[:, self.ee_body_id]
      # Compute L2 distance for position
      pos_error = torch.norm(self.target_pos - ee_pos, dim=-1)
      self.metrics["position_error"] += pos_error
      
      ee_quat = self.robot.data.body_link_quat_w[:, self.ee_body_id]
      # Compute quaternion distance (1 - |q1 . q2|) which is bounded [0, 1]
      quat_dot = torch.abs(torch.sum(self.target_quat * ee_quat, dim=-1))
      ori_error = 1.0 - quat_dot
      self.metrics["orientation_error"] += ori_error

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

    if getattr(self.cfg, "mode", "absolute") == "relative" and self.ee_body_id != -1:
      is_first_step = (self.command_counter[env_ids] == 0)
      
      base_pos = torch.where(
        is_first_step.unsqueeze(-1),
        self.robot.data.body_link_pos_w[env_ids, self.ee_body_id],
        self.target_pos[env_ids]
      )
      base_quat = torch.where(
        is_first_step.unsqueeze(-1),
        self.robot.data.body_link_quat_w[env_ids, self.ee_body_id],
        self.target_quat[env_ids]
      )
      
      target_pos = base_pos + target_pos
      target_quat = quat_mul(target_quat, base_quat)
      
      # Clamp to workspace bounds to prevent drifting away infinitely
      bounds = getattr(self.cfg, "workspace", None)
      if bounds is not None:
        target_pos[..., 0] = torch.clamp(target_pos[..., 0], bounds.pos_x[0], bounds.pos_x[1])
        target_pos[..., 1] = torch.clamp(target_pos[..., 1], bounds.pos_y[0], bounds.pos_y[1])
        target_pos[..., 2] = torch.clamp(target_pos[..., 2], bounds.pos_z[0], bounds.pos_z[1])

    self.target_pos[env_ids] = target_pos
    self.target_quat[env_ids] = target_quat

    if self.target_mocap_id != -1:
      self.robot.data.data.mocap_pos[env_ids, self.target_mocap_id] = target_pos
      self.robot.data.data.mocap_quat[env_ids, self.target_mocap_id] = target_quat

  def _update_command(self) -> None:
    if self.target_mocap_id != -1:
      pos = self.robot.data.data.mocap_pos[:, self.target_mocap_id].clone()
      bounds = getattr(self.cfg, "workspace", None)
      if bounds is not None:
        pos[..., 0] = torch.clamp(pos[..., 0], bounds.pos_x[0], bounds.pos_x[1])
        pos[..., 1] = torch.clamp(pos[..., 1], bounds.pos_y[0], bounds.pos_y[1])
        pos[..., 2] = torch.clamp(pos[..., 2], bounds.pos_z[0], bounds.pos_z[1])
        # Write clamped pos back to sim so the GUI visualizer snaps back
        self.robot.data.data.mocap_pos[:, self.target_mocap_id] = pos

      self.target_pos[:] = pos
      self.target_quat[:] = self.robot.data.data.mocap_quat[:, self.target_mocap_id]

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
      
    import numpy as np
    def quat_to_rot_mat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])

    for batch in env_indices:
      target_pos = self.target_pos[batch].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.03,
        color=(0.0, 1.0, 0.0, 0.5),
        label=f"target_position_{batch}",
      )
      
      target_quat = self.target_quat[batch].cpu().numpy()
      visualizer.add_frame(
        position=target_pos,
        rotation_matrix=quat_to_rot_mat(target_quat),
        scale=0.2,
        axis_radius=0.01,
        label=f"target_frame_{batch}",
      )
      
      if self.ee_body_id != -1:
        ee_pos = self.robot.data.body_link_pos_w[batch, self.ee_body_id].cpu().numpy()
        ee_quat = self.robot.data.body_link_quat_w[batch, self.ee_body_id].cpu().numpy()
        visualizer.add_frame(
          position=ee_pos,
          rotation_matrix=quat_to_rot_mat(ee_quat),
          scale=0.2,
          axis_radius=0.01,
          label=f"ee_frame_{batch}",
        )

@dataclass(kw_only=True)
class UserPoseCommandCfg(CommandTermCfg):
  asset_name: str = "robot"
  body_name: str = "ee_link"
  mode: Literal["absolute", "relative"] = "absolute"
  
  @dataclass
  class Ranges:
    pos_x: tuple[float, float] = (0.2, 0.6)
    pos_y: tuple[float, float] = (-0.5, 0.5)
    pos_z: tuple[float, float] = (0.3, 0.8)
    roll: tuple[float, float] = (-1.57, 1.57)
    pitch: tuple[float, float] = (-1.57, 1.57)
    yaw: tuple[float, float] = (-1.57, 1.57)

  @dataclass
  class WorkspaceBounds:
    pos_x: tuple[float, float] = (0.2, 0.6)
    pos_y: tuple[float, float] = (-0.5, 0.5)
    pos_z: tuple[float, float] = (0.3, 0.8)

  ranges: Ranges = field(default_factory=Ranges)
  workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)

  def build(self, env: ManagerBasedRlEnv) -> UserPoseCommand:
    return UserPoseCommand(self, env)
