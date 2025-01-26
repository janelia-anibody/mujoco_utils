"""Utilities for working with and manipulating MJCF models."""

from typing import Sequence
from lxml import etree

import numpy as np
from dm_control import mjcf

from mujoco_utils.quaternions import (
    conj_quat,
    mult_quat,
    rotate_vec_with_quat,
)


def mjcf2xml(mjcf_model: mjcf.RootElement,
             output_file_name: str | None = None,
             precision: int = 5,
             zero_threshold: float = 1e-7,
            ) -> str | None:
    """Generate clean XML string from mjcf_model and optionally save to file.

    Args:
        mjcf_model: MJCF model to export.
        output_file_name: Optional file name, if provided, the generated XML
            string be saved to this file.
        precision: Number of digits to output for floating point quantities.
        zero_threshold: When outputting XML, floating point quantities whose
            absolute value falls below this threshold will be treated as zero.

    Returns:
        XML string or, if `output_file_name` is provided, saves XML string to file.
    """
    
    # Dirty export to string first.
    xml_string = mjcf_model.to_xml_string(
        'float',
        precision=precision,
        zero_threshold=zero_threshold,
        filename_with_hash=False,
    )
    
    # Remove empty default.
    root = etree.XML(xml_string, etree.XMLParser(remove_blank_text=True))
    default_elem = root.find('default')
    root.insert(3, default_elem[0])
    root.remove(default_elem)

    # Get string from lxml and remove class="/".
    xml_string = etree.tostring(root, pretty_print=True)
    xml_string = xml_string.replace(b' class="/"', b'')

    # Remove gravcomp="0".
    xml_string = xml_string.replace(b' gravcomp="0"', b'')

    # Insert spaces between top level elements.
    lines = xml_string.splitlines()
    newlines = []
    for line in lines:
        newlines.append(line)
        if line.startswith(b'  <'):
            if line.startswith(b'  </') or line.endswith(b'/>'):
                newlines.append(b'')
    newlines.append(b'')
    xml_string = b'\n'.join(newlines)

    # Save generated XML string to file or return XML string.
    if output_file_name is not None:
        with open(output_file_name, 'wb') as f:
            f.write(xml_string)
    else:
        return xml_string


def change_body_frame(body: mjcf.Element,
                      frame_pos: Sequence | None = None,
                      frame_quat: Sequence | None = None):
    """In-place change the frame of a body while maintaining child locations."""
    frame_pos = np.zeros(3) if frame_pos is None else frame_pos
    frame_quat = np.array((1., 0, 0, 0)) if frame_quat is None else frame_quat
    # Get frame transformation.
    body_pos = np.zeros(3) if body.pos is None else body.pos
    dpos = body_pos - frame_pos
    body_quat = np.array((1., 0, 0, 0)) if body.quat is None else body.quat
    dquat = mult_quat(conj_quat(frame_quat), body_quat)
    # Translate and rotate the body to the new frame.
    body.pos = frame_pos
    body.quat = frame_quat
    # Move all its children to their previous location.
    for child in body.all_children():
        if not hasattr(child, 'pos'):
            continue
        # Rotate:
        if hasattr(child, 'quat'):
            child_quat = np.array(
                (1., 0, 0, 0)) if child.quat is None else child.quat
            child.quat = mult_quat(dquat, child_quat)
        # Translate, accounting for rotations.
        child_pos = np.zeros(3) if child.pos is None else child.pos
        pos_in_parent = rotate_vec_with_quat(child_pos, body_quat) + dpos
        child.pos = rotate_vec_with_quat(pos_in_parent, conj_quat(frame_quat))


def get_mjcf_tree(element: mjcf.Element,
                  bodies_only: bool = False) -> dict:
    """Returns dict representing kinematic tree of mjcf model.
    
    Example:
        tree = get_mjcf_tree(mjcf_model.worldbody)
        print_tree(tree)
    """
    tree = {}
    if not element._children:
        return ''
    if bodies_only:
        n_bodies = len([child for child in element._children 
                        if child.tag == 'body'])
        if n_bodies == 0:
            return ''
    for child in element._children:
        if bodies_only and child.tag != 'body':
            continue
        tree[f'{child.tag}: {child.name}'] = get_mjcf_tree(child, bodies_only)
    return tree
