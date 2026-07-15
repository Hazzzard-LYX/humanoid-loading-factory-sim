#!/usr/bin/env python3
"""Small URDF FK/IK helper for the single-robot loading prototype."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT_DIR = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT_DIR / "biped_s62" / "urdf" / "biped_s62.urdf"

RIGHT_ACTIVE_JOINTS = [
    "knee_joint",
    "leg_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
    *[f"zarm_r{index}_joint" for index in range(1, 8)],
]

COMMON_ACTIVE_JOINTS = [
    "knee_joint",
    "leg_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
]
BIMANUAL_ACTIVE_JOINTS = [
    *COMMON_ACTIVE_JOINTS,
    *[f"zarm_l{index}_joint" for index in range(1, 8)],
    *[f"zarm_r{index}_joint" for index in range(1, 8)],
]


def xyz(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array([float(value) for value in text.split()], dtype=float) if text else np.array(default, dtype=float)


def transform(rotation=np.eye(3), translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    return Rotation.from_euler("xyz", rpy).as_matrix()


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class RobotKinematics:
    def __init__(self, urdf_path: Path = URDF_PATH) -> None:
        root = ET.parse(urdf_path).getroot()
        self.joints: dict[str, Joint] = {}
        self.parent_joint_by_child: dict[str, Joint] = {}
        for element in root.findall("joint"):
            origin_element = element.find("origin")
            origin_xyz = xyz(origin_element.get("xyz") if origin_element is not None else None)
            origin_rpy = xyz(origin_element.get("rpy") if origin_element is not None else None)
            axis_element = element.find("axis")
            axis = xyz(axis_element.get("xyz") if axis_element is not None else None, (1.0, 0.0, 0.0))
            limit = element.find("limit")
            joint_type = element.get("type", "fixed")
            if joint_type == "continuous":
                lower, upper = -math.pi, math.pi
            elif limit is not None:
                lower = float(limit.get("lower", "0"))
                upper = float(limit.get("upper", "0"))
            else:
                lower = upper = 0.0
            joint = Joint(
                name=element.get("name"),
                joint_type=joint_type,
                parent=element.find("parent").get("link"),
                child=element.find("child").get("link"),
                origin=transform(rpy_matrix(origin_rpy), origin_xyz),
                axis=axis / np.linalg.norm(axis),
                lower=lower,
                upper=upper,
            )
            self.joints[joint.name] = joint
            self.parent_joint_by_child[joint.child] = joint

    def chain(self, target_link: str, root_link: str = "base_link") -> list[Joint]:
        chain = []
        link = target_link
        while link != root_link:
            joint = self.parent_joint_by_child[link]
            chain.append(joint)
            link = joint.parent
        chain.reverse()
        return chain

    @staticmethod
    def base_transform(base_pose: np.ndarray) -> np.ndarray:
        x, y, yaw = base_pose
        return transform(Rotation.from_euler("z", yaw).as_matrix(), (x, y, 0.0))

    def fk(
        self,
        target_link: str,
        base_pose: np.ndarray,
        joint_positions: dict[str, float],
    ) -> np.ndarray:
        value = self.base_transform(base_pose)
        for joint in self.chain(target_link):
            value = value @ joint.origin
            q = joint_positions.get(joint.name, 0.0)
            if joint.joint_type in ("revolute", "continuous"):
                value = value @ transform(Rotation.from_rotvec(joint.axis * q).as_matrix())
            elif joint.joint_type == "prismatic":
                value = value @ transform(translation=joint.axis * q)
        return value

    def active_bounds(self, names=RIGHT_ACTIVE_JOINTS) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([self.joints[name].lower for name in names], dtype=float)
        upper = np.array([self.joints[name].upper for name in names], dtype=float)
        return lower, upper

    def solve_right_finger(
        self,
        base_pose: np.ndarray,
        target_position: np.ndarray,
        target_finger_z: np.ndarray,
        seed: np.ndarray,
        reference: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float | bool]]:
        names = RIGHT_ACTIVE_JOINTS
        lower, upper = self.active_bounds(names)
        # Keep a deliberate margin from hard stops so interpolation and small
        # numerical corrections cannot drive a joint into its URDF limit.
        solve_lower = lower + math.radians(2.0)
        solve_upper = upper - math.radians(2.0)
        seed = np.clip(np.asarray(seed, dtype=float), solve_lower, solve_upper)
        reference = seed.copy() if reference is None else np.asarray(reference, dtype=float)
        target_finger_z = np.asarray(target_finger_z, dtype=float)
        target_finger_z /= np.linalg.norm(target_finger_z)
        point_local = np.array([0.008307, -0.0015, 0.068487, 1.0])
        ranges = np.maximum(upper - lower, 0.1)

        def residual(q: np.ndarray) -> np.ndarray:
            joint_map = dict(zip(names, q))
            finger = self.fk("zarm_r7_finger_link", base_pose, joint_map)
            position = (finger @ point_local)[:3]
            finger_z = finger[:3, 2]
            position_error = (position - target_position) / 0.015
            axis_error = np.cross(finger_z, target_finger_z) / 0.035
            regularization = 0.08 * (q - reference) / ranges
            return np.concatenate((position_error, axis_error, regularization))

        result = least_squares(
            residual,
            seed,
            bounds=(solve_lower, solve_upper),
            max_nfev=2500,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        joint_map = dict(zip(names, result.x))
        finger = self.fk("zarm_r7_finger_link", base_pose, joint_map)
        solved_position = (finger @ point_local)[:3]
        solved_axis = finger[:3, 2]
        info = {
            "success": bool(result.success),
            "position_error": float(np.linalg.norm(solved_position - target_position)),
            "axis_error_deg": float(
                math.degrees(
                    math.acos(np.clip(np.dot(solved_axis, target_finger_z), -1.0, 1.0))
                )
            ),
            "cost": float(result.cost),
        }
        return result.x, info

    def solve_bimanual_fingers(
        self,
        base_pose: np.ndarray,
        left_position: np.ndarray,
        left_finger_z: np.ndarray,
        right_position: np.ndarray,
        right_finger_z: np.ndarray,
        seed: np.ndarray,
        reference: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float | bool]]:
        """Solve both fingertips together while sharing the waist/leg joints."""
        names = BIMANUAL_ACTIVE_JOINTS
        lower, upper = self.active_bounds(names)
        solve_lower = lower + math.radians(2.0)
        solve_upper = upper - math.radians(2.0)
        seed = np.clip(np.asarray(seed, dtype=float), solve_lower, solve_upper)
        reference = seed.copy() if reference is None else np.asarray(reference, dtype=float)
        left_finger_z = np.asarray(left_finger_z, dtype=float)
        right_finger_z = np.asarray(right_finger_z, dtype=float)
        left_finger_z /= np.linalg.norm(left_finger_z)
        right_finger_z /= np.linalg.norm(right_finger_z)
        point_local = np.array([0.008307, -0.0015, 0.068487, 1.0])
        ranges = np.maximum(upper - lower, 0.1)

        def finger_state(side: str, joint_map: dict[str, float]):
            finger = self.fk(f"zarm_{side}7_finger_link", base_pose, joint_map)
            return (finger @ point_local)[:3], finger[:3, 2]

        def residual(q: np.ndarray) -> np.ndarray:
            joint_map = dict(zip(names, q))
            left_solved_position, left_solved_axis = finger_state("l", joint_map)
            right_solved_position, right_solved_axis = finger_state("r", joint_map)
            return np.concatenate(
                (
                    (left_solved_position - left_position) / 0.015,
                    (left_solved_axis - left_finger_z) / 0.070,
                    (right_solved_position - right_position) / 0.015,
                    (right_solved_axis - right_finger_z) / 0.070,
                    0.06 * (q - reference) / ranges,
                )
            )

        result = least_squares(
            residual,
            seed,
            bounds=(solve_lower, solve_upper),
            max_nfev=5000,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        joint_map = dict(zip(names, result.x))
        left_solved_position, left_solved_axis = finger_state("l", joint_map)
        right_solved_position, right_solved_axis = finger_state("r", joint_map)
        info = {
            "success": bool(result.success),
            "left_position_error": float(np.linalg.norm(left_solved_position - left_position)),
            "right_position_error": float(np.linalg.norm(right_solved_position - right_position)),
            "left_axis_error_deg": float(
                math.degrees(math.acos(np.clip(np.dot(left_solved_axis, left_finger_z), -1.0, 1.0)))
            ),
            "right_axis_error_deg": float(
                math.degrees(math.acos(np.clip(np.dot(right_solved_axis, right_finger_z), -1.0, 1.0)))
            ),
            "cost": float(result.cost),
        }
        return result.x, info


def default_seed(kinematics: RobotKinematics) -> np.ndarray:
    preferred = {
        "knee_joint": 0.35,
        "leg_joint": -0.75,
        "waist_pitch_joint": 0.40,
        "waist_yaw_joint": 0.0,
        "zarm_r1_joint": -0.35,
        "zarm_r2_joint": -0.25,
        "zarm_r3_joint": 0.0,
        "zarm_r4_joint": -1.10,
        "zarm_r5_joint": 0.0,
        "zarm_r6_joint": 0.0,
        "zarm_r7_joint": 0.0,
    }
    lower, upper = kinematics.active_bounds()
    values = np.array([preferred[name] for name in RIGHT_ACTIVE_JOINTS])
    return np.clip(values, lower + 1e-4, upper - 1e-4)


def default_bimanual_seed(kinematics: RobotKinematics) -> np.ndarray:
    preferred = {
        "knee_joint": 0.35,
        "leg_joint": -0.75,
        "waist_pitch_joint": 0.40,
        "waist_yaw_joint": 0.0,
        "zarm_l1_joint": -0.35,
        "zarm_l2_joint": 0.25,
        "zarm_l3_joint": 0.0,
        "zarm_l4_joint": -1.10,
        "zarm_l5_joint": 0.0,
        "zarm_l6_joint": 0.0,
        "zarm_l7_joint": 0.0,
        "zarm_r1_joint": -0.35,
        "zarm_r2_joint": -0.25,
        "zarm_r3_joint": 0.0,
        "zarm_r4_joint": -1.10,
        "zarm_r5_joint": 0.0,
        "zarm_r6_joint": 0.0,
        "zarm_r7_joint": 0.0,
    }
    lower, upper = kinematics.active_bounds(BIMANUAL_ACTIVE_JOINTS)
    values = np.array([preferred[name] for name in BIMANUAL_ACTIVE_JOINTS])
    return np.clip(values, lower + 1e-4, upper - 1e-4)


if __name__ == "__main__":
    kin = RobotKinematics()
    print("active joints:")
    for name, value in zip(RIGHT_ACTIVE_JOINTS, default_seed(kin)):
        joint = kin.joints[name]
        print(f"  {name}: seed={value:.4f}, limits=({joint.lower:.4f}, {joint.upper:.4f})")
