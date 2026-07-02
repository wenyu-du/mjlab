import torch
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_box_minus, quat_mul, quat_conjugate, matrix_from_quat

def position_error_tanh(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg, weight: float = 1.0) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    position_cmd = command[:, :3]
    # use site_pos_w for ee_link or body_link_pos_w
    if asset_cfg.site_ids:
        position = asset.data.site_pos_w[:, asset_cfg.site_ids, :].squeeze(1)
    else:
        position = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :].squeeze(1)
    error = position_cmd - position
    magnitude = torch.linalg.norm(error, dim=-1, ord=2, keepdim=True)
    direction = torch.where(magnitude > 1e-8, error / magnitude, torch.zeros_like(error))
    torch.tanh_(magnitude)
    return weight * magnitude * direction

def orientation_error_matrix(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    orientation_cmd = command[:, 3:]
    if asset_cfg.site_ids:
        orientation = asset.data.site_quat_w[:, asset_cfg.site_ids, :].squeeze(1)
    else:
        orientation = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    error = quat_mul(orientation_cmd, quat_conjugate(orientation))
    mat = matrix_from_quat(error)
    return mat[..., :2].reshape(env.num_envs, -1)

def motion_position_error_exp(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    position_cmd = env.command_manager.get_command(command_name)[:, :3]
    if asset_cfg.site_ids:
        position = asset.data.site_pos_w[:, asset_cfg.site_ids, :].squeeze(1)
    else:
        position = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :].squeeze(1)
    error = torch.sum(torch.square(position_cmd - position), dim=-1)
    return torch.exp(-error / std**2)

def motion_position_error_lin(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    position_cmd = env.command_manager.get_command(command_name)[:, :3]
    if asset_cfg.site_ids:
        position = asset.data.site_pos_w[:, asset_cfg.site_ids, :].squeeze(1)
    else:
        position = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :].squeeze(1)
    error = torch.sum(torch.square(position_cmd - position), dim=-1)
    return error

def motion_orientation_error_exp(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    orientation_cmd = env.command_manager.get_command(command_name)[:, 3:]
    if asset_cfg.site_ids:
        orientation = asset.data.site_quat_w[:, asset_cfg.site_ids, :].squeeze(1)
    else:
        orientation = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    error = torch.sum(torch.square(quat_box_minus(orientation_cmd, orientation)), dim=-1)
    return torch.exp(-error / std**2)

def motion_orientation_error_lin(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    orientation_cmd = env.command_manager.get_command(command_name)[:, 3:]
    if asset_cfg.site_ids:
        orientation = asset.data.site_quat_w[:, asset_cfg.site_ids, :].squeeze(1)
    else:
        orientation = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    error = torch.sum(torch.square(quat_box_minus(orientation_cmd, orientation)), dim=-1)
    return error

def motion_position_distance(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    position_cmd = env.command_manager.get_command(command_name)[:, :3]
    if asset_cfg.site_ids:
        position = asset.data.site_pos_w[:, asset_cfg.site_ids, :].squeeze(1)
    else:
        position = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :].squeeze(1)
    return torch.linalg.norm(position_cmd - position, dim=-1)

def motion_orientation_distance(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    orientation_cmd = env.command_manager.get_command(command_name)[:, 3:]
    if asset_cfg.site_ids:
        orientation = asset.data.site_quat_w[:, asset_cfg.site_ids, :].squeeze(1)
    else:
        orientation = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    rot_vec = quat_box_minus(orientation_cmd, orientation)
    return torch.linalg.norm(rot_vec, dim=-1)
