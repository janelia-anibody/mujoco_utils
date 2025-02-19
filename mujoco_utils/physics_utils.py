"""Utilities for compiled mjcf.Physics."""

from typing import Sequence

import numpy as np
from dm_control import mjcf

from mujoco_utils.quaternions import (
    rotate_vec_with_quat,
    reciprocal_quat,
)


def site_pos_in_body_frame(physics: mjcf.Physics,
                           body_name: str,
                           site_xpos: Sequence | None = None,
                           site_name: str | None = None,
                          ) -> np.ndarray:
    """Get site position in local reference frame of given body.

    The site can be child of any body, not only the worldbody.

    Args:
        physics: mjcf.Physics instance.
        body_name: Site coordinates will be returned in the local reference frame
            of this body.
        site_xpos: 3-tuple, site position in world coordinates. If site_xpos is
            provided, site_name is ignored. Either site_xpos or site_name
            must be provided.
        site_name: Name of site to get the position of. If site_xpos is
            provided, site_name is ignored.
    Returns:
        Position of site in body's local frame, as if the site was child of this
            body in the first place.
    """
    if site_xpos is None:
        site_xpos = physics.named.data.site_xpos[site_name]
    quat = physics.named.data.xquat[body_name]
    xpos = physics.named.data.xpos[body_name]
    pos = rotate_vec_with_quat(site_xpos - xpos,
                               reciprocal_quat(quat))
    return pos


def joint_to_dof_id(physics: mjcf.Physics,
                    joint_name: str = None,
                    joint_id: int = None) -> int | list[int]:
    """Get degree-of-freedom index dof_id from joint_name or joint_id.
    
    dof_id can be used to index into qvel, qfrc_applied, qfrc_actuator, etc.

    Args:
        physics: An mjcf.physics instance.
        joint_name, joint_id: Either name or id of joint to get dof_id for.
            If both are provided, joint_id is ignored.

    Returns:
        dof_id for requested joint. For freejoint or slider, returns list of ids.
    """
    if joint_name is None and joint_id is None:
        raise ValueError('Either joint_name or joint_id should be provided.')
    if joint_name is not None:
        joint_id = physics.model.name2id(joint_name, 'joint')
    jnt_type = physics.model.jnt_type[joint_id]
    dof_id = physics.model.jnt_dofadr[joint_id]
    # Check if free or ball joint. Otherwise, slide and hidge have only one DoF.
    if jnt_type == 0:
        # Free joint.
        dof_id = [*range(dof_id, dof_id+6)]
    elif jnt_type == 1:
        # Ball joint.
        dof_id = [*range(dof_id, dof_id+3)]
    return dof_id


def is_position_actuator(physics: mjcf.Physics,
                         actuator_id: int) -> bool:
    """Check if given actuator a position actuator parametrized as:
    biastype: "affine"
    gainprm: (kp, 0, 0)
    biasprm: (0, -kp, 0)    
    TODO: later could also consider biasprm: (0, -kp, -kv)
    """
    biastype = physics.model.actuator_biastype[actuator_id]
    gainprm = physics.model.actuator_gainprm[actuator_id]
    biasprm = physics.model.actuator_biasprm[actuator_id]
    print(gainprm[0], biasprm[1])
    return (biastype == 1 and
            np.isclose(gainprm[0], - biasprm[1]) and
            all(gainprm[1:] == 0) and
            biasprm[0] == 0 and
            all(biasprm[2:] == 0))


def get_enabled_observables(walker) -> dict:
    """Get dict of enabled observables from walker."""
    enabled_obs = {}
    for k, v in walker.observables._observables.items():
        if v.enabled:
            enabled_obs[k] = v
    return enabled_obs


def get_critical_damping(physics: mjcf.Physics,
                         joint_name: str,
                         actuator_name: str | None = None,
                         joint_spring: bool = True,
                         actuator_spring: bool = True) -> float:
    """Calculate critical damping for a joint, possibly taking into account both
    actuator gainprm and joint stiffness spring.
    
    Args:
        physics: A physics instance.
        joint_name: Joint name to calculate critical damping for.
        actuator_name: Actuator name if different from joint_name.
        joint_spring: Whether to use joint (stiffness) spring constant.
        actuator_spring: Whether to use actuator gainprm as spring constant.
        
    Returns:
        Critical damping.
    """
    if actuator_name is None:
        actuator_name = joint_name

    joint_id = physics.named.model.jnt_qposadr[joint_name]
    inertia = physics.model.dof_M0[joint_id]

    spring_const = 0.
    if joint_spring:
        spring_const = spring_const + physics.named.model.jnt_stiffness[joint_name]
    if actuator_spring:
        spring_const = (spring_const + 
                        physics.named.model.actuator_gainprm[actuator_name][0])

    critical_damping = 2 * np.sqrt(spring_const * inertia)
    return critical_damping
