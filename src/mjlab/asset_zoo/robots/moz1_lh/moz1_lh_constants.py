"""MOZ1 Left Hand constants."""

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

MOZ1_LH_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "moz1_lh" / "xmls" / "left_hand.xml"
)
assert MOZ1_LH_XML.exists()

def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(MOZ1_LH_XML))

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.5),
  joint_pos={
    "j_lh_.": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=("j_lh_.*",)),),
  soft_joint_pos_limit_factor=0.95,
)

def get_moz1_lh_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )
