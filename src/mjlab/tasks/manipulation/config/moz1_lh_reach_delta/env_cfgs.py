import math
from typing import Any

import torch
from mjlab.asset_zoo.robots.moz1_lh.moz1_lh_constants import get_moz1_lh_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, ManagerBasedRlEnv
from mjlab.envs.mdp import (
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  time_out,
  reset_joints_by_offset,
  joint_acc_l2,
  action_acc_l2,
)
from mjlab.envs.mdp.rewards import joint_vel_l2, action_rate_l2, joint_pos_limits, joint_torques_l2
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  EventTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.tasks.manipulation.mdp.user_pose_command import UserPoseCommandCfg

from . import mdp

def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.square(env.action_manager.action), dim=-1)

def moz1_lh_reach_delta_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  
  # Entity config
  robot_cfg = get_moz1_lh_robot_cfg()

  # Actions
  actions = {
    "joint_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=("j_lh_1", "j_lh_2", "j_lh_3", "j_lh_4", "j_lh_5", "j_lh_6", "j_lh_7"),
      scale=0.06,
      preserve_order=True,
    )
  }
  
  # Commands
  commands = {
    "pose_cmd": UserPoseCommandCfg(
      asset_name="robot",
      body_name="ee_link",
      mode="relative",
      debug_vis=True,
      resampling_time_range=(0.033, 0.033),
      ranges=UserPoseCommandCfg.Ranges(
        pos_x=(-0.005, 0.005),
        pos_y=(-0.005, 0.005),
        pos_z=(-0.005, 0.005),
        roll=(-0.035, 0.035),
        pitch=(-0.035, 0.035),
        yaw=(-0.035, 0.035),
      ),
      workspace=UserPoseCommandCfg.WorkspaceBounds(
        pos_x=(0.2, 0.55),
        pos_y=(-0.4, 0.4),
        pos_z=(0.3, 0.7),
      )
    )
  }

  # Observations
  actor_terms = {
    "pos_err_tanh": ObservationTermCfg(
      func=mdp.position_error_tanh,
      params={"command_name": "pose_cmd", "weight": 10.0, "asset_cfg": SceneEntityCfg("robot", body_names=("ee_link",))}
    ),
    "ori_err_matrix": ObservationTermCfg(
      func=mdp.orientation_error_matrix,
      params={"command_name": "pose_cmd", "asset_cfg": SceneEntityCfg("robot", body_names=("ee_link",))}
    ),
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("j_lh_1", "j_lh_2", "j_lh_3", "j_lh_4", "j_lh_5", "j_lh_6", "j_lh_7"))},
      noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01) if not play else None,
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("j_lh_1", "j_lh_2", "j_lh_3", "j_lh_4", "j_lh_5", "j_lh_6", "j_lh_7"))},
      noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1) if not play else None,
    ),
    "last_action": ObservationTermCfg(func=last_action),
  }
  
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True, concatenate_terms=True, history_length=2),
    "critic": ObservationGroupCfg(actor_terms, enable_corruption=False, concatenate_terms=True, history_length=2),
  }

  # Events
  events = {
    "reset_joint_pos": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.8, 0.8),
        "velocity_range": (-0.1, 0.1),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("j_lh_.*",)),
      }
    )
  }

  # Rewards
  rewards = {
    "track_position_far": RewardTermCfg(
      func=mdp.motion_position_error_exp,
      weight=10.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("ee_link",)), "command_name": "pose_cmd", "std": 0.25}
    ),
    "track_position_near": RewardTermCfg(
      func=mdp.motion_position_error_exp,
      weight=20.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("ee_link",)), "command_name": "pose_cmd", "std": 0.02}
    ),
    "track_orientation_far": RewardTermCfg(
      func=mdp.motion_orientation_error_exp,
      weight=10.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("ee_link",)), "command_name": "pose_cmd", "std": 0.25}
    ),
    "track_orientation_near": RewardTermCfg(
      func=mdp.motion_orientation_error_exp,
      weight=15.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("ee_link",)), "command_name": "pose_cmd", "std": 0.05}
    ),
    "track_position_distance": RewardTermCfg(
      func=mdp.motion_position_distance,
      weight=-100.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("ee_link",)), "command_name": "pose_cmd"}
    ),
    "track_orientation_distance": RewardTermCfg(
      func=mdp.motion_orientation_distance,
      weight=-50.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("ee_link",)), "command_name": "pose_cmd"}
    ),
    "joint_vel_penalty": RewardTermCfg(func=joint_vel_l2, weight=-0.05),
    "joint_acc_penalty": RewardTermCfg(func=joint_acc_l2, weight=-1e-4),
    "action_l2_penalty": RewardTermCfg(func=action_l2, weight=-0.02),
    "action_rate_penalty": RewardTermCfg(func=action_rate_l2, weight=-0.02),
    "action_acc_penalty": RewardTermCfg(func=action_acc_l2, weight=-0.005),
    "joint_pos_limits_penalty": RewardTermCfg(func=joint_pos_limits, weight=-10.0),
    "joint_torques_penalty": RewardTermCfg(func=joint_torques_l2, weight=-1e-4),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
      num_envs=4096 if not play else 1,
      env_spacing=3.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=2.0,
      elevation=-15.0,
      azimuth=150.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.005, disableflags=("contact",)),
    ),
    decimation=4,
    episode_length_s=10.0 if not play else 60.0,
  )

  return cfg
