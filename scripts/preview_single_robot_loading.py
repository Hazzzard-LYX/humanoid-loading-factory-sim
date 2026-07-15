#!/usr/bin/env python3
"""Preview one humanoid loading two tubes into the first machine.

This is deliberately a deterministic kinematic preview.  The robot base and
joints are interpolated through collision-conscious keyframes.  Each hand
extracts one tube and inserts it into one of the fixture's two openings.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT_DIR = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT_DIR / "scenes" / "humanoid_loading_factory.usd"
TUBE_ASSET_PATH = ROOT_DIR / "assets" / "materials" / "aluminum_tube.usd"

ROBOT_STATION_PATH = "/World/Factory/Robots/Robot_01_West_Pair_01"
ROBOT_PATH = f"{ROBOT_STATION_PATH}/Model"
MACHINE_1_PRESS_PATH = (
    "/World/Factory/Machines/Bank_01_West/Cell_01/"
    "Machine_01_Wall_Process01/Press"
)
MACHINE_2_PRESS_PATH = (
    "/World/Factory/Machines/Bank_01_West/Cell_01/"
    "Machine_02_Aisle_Process02/Press"
)
SOURCE_TUBES_ROOT = (
    "/World/Factory/MaterialRacks/Bank_01_West/Cell_01/"
    "RawMaterialRack_Machine01_MirroredSide/Bins/Bin_01/"
    "WorldYawAlignment/AisleAxisTilt/Orientation/Tubes"
)
SOURCE_TUBE_PATHS = {
    "left": f"{SOURCE_TUBES_ROOT}/Tube_R01_C01",
    "right": f"{SOURCE_TUBES_ROOT}/Tube_R01_C10",
}
CARRIED_TUBE_PATHS = {
    "left": "/World/SingleRobotLoadingDemo/CarriedTubeLeft",
    "right": "/World/SingleRobotLoadingDemo/CarriedTubeRight",
}

HOME_BASE = np.array([-6.215, -9.065, math.radians(225.0)])
PICKUP_BASE = np.array([-7.130, -7.900, math.radians(-155.0)])
# Keep the mobile base farther from the machine face.  At the previous X=-6.35
# transfer pose the pickup-arm posture already overlapped the press top before
# the reorientation phase even began.
MACHINE_BASE = np.array([-6.100, -9.065, math.pi])
MACHINE_2_BASE = np.array([-6.215, -9.150, -math.pi / 2.0])
FINISHED_ROUTE_CLEAR_BASE = np.array([-5.000, -8.650, math.radians(-135.0)])
FINISHED_BIN_BASE = np.array([-4.600, -9.850, math.radians(-135.0)])

TUBE_OPEN_END = {
    "left": np.array([-7.5975, -8.2825, 1.1770]),
    "right": np.array([-8.0025, -8.2825, 1.1770]),
}


BIN_CENTER_PITCH_Y = 0.28


def raw_pair_spec(
    row: int,
    left_column: int,
    right_column: int,
    bin_index: int = 1,
):
    if bin_index not in (1, 2):
        raise ValueError(f"bin_index must be 1 or 2, got {bin_index}")
    bin_y_offset = BIN_CENTER_PITCH_Y * (bin_index - 1)

    def open_end(column: int):
        return np.array(
            [
                -7.5975 - 0.045 * (column - 1),
                -8.2825 + bin_y_offset + 0.045 * (row - 1),
                1.1770,
            ]
        )

    def finished_open_end(column: int):
        return np.array(
            [
                -5.0775 - 0.045 * (column - 1),
                -10.5025 + bin_y_offset + 0.045 * (row - 1),
                1.1770,
            ]
        )

    return {
        "bin_index": bin_index,
        "label": (
            f"B{bin_index:02d}:R{row:02d}:"
            f"C{left_column:02d}+C{right_column:02d}"
        ),
        "paths": {
            "left": (
                f"{SOURCE_TUBES_ROOT.replace('/Bin_01/', f'/Bin_{bin_index:02d}/')}"
                f"/Tube_R{row:02d}_C{left_column:02d}"
            ),
            "right": (
                f"{SOURCE_TUBES_ROOT.replace('/Bin_01/', f'/Bin_{bin_index:02d}/')}"
                f"/Tube_R{row:02d}_C{right_column:02d}"
            ),
        },
        "open_ends": {
            "left": open_end(left_column),
            "right": open_end(right_column),
        },
        "finished_open_ends": {
            "left": finished_open_end(left_column),
            "right": finished_open_end(right_column),
        },
        "pickup_base": np.array(
            [
                -7.130,
                -7.900 + bin_y_offset + 0.045 * (row - 1),
                math.radians(-155.0),
            ]
        ),
        "finished_base": np.array(
            [
                -4.600,
                -9.850 + bin_y_offset + 0.045 * (row - 1),
                math.radians(-135.0),
            ]
        ),
    }


CONTINUOUS_PICKUP_SCHEDULE = []
for _bin_index in (1, 2):
    for _row in (1, 2, 5, 6):
        for _left_column, _right_column in (
            (1, 10),
            (2, 9),
            (3, 6),
            (4, 8),
            (5, 7),
        ):
            CONTINUOUS_PICKUP_SCHEDULE.append(
                raw_pair_spec(
                    _row,
                    _left_column,
                    _right_column,
                    bin_index=_bin_index,
                )
            )
    for _row in (3, 4):
        for _left_column, _right_column in ((1, 10), (2, 9)):
            CONTINUOUS_PICKUP_SCHEDULE.append(
                raw_pair_spec(
                    _row,
                    _left_column,
                    _right_column,
                    bin_index=_bin_index,
                )
            )
# The actual loading fixture center was established by the accepted single-
# tube preview.  The higher closed loops in crush.STL belong to the machine
# housing and are not the tube seats.
MACHINE_HOLE_WORLD_Z = 0.970775
# The usable tube seats lie near the inner edges of the two fixture windows,
# not at the window centroids.  Keep the two tube axes 90 mm apart around the
# machine centerline.
MACHINE_HOLE_HALF_SPACING = 0.045
MACHINE_HOLE_WORLD_Y = {
    "left": -9.065 - MACHINE_HOLE_HALF_SPACING,
    "right": -9.065 + MACHINE_HOLE_HALF_SPACING,
}
MACHINE_2_HOLE_WORLD_X = {"left": -6.170, "right": -6.260}
FINISHED_TUBE_OPEN_END = {
    "left": np.array([-5.0775, -10.5025, 1.1770]),
    "right": np.array([-5.4825, -10.5025, 1.1770]),
}
PRESS_RAISED_Z = 0.200
PRESS_DOWN_Z = 0.0
TUBE_LENGTH = 0.452
FINGER_INSERTION = 0.018
FINGER_TO_TUBE_START = TUBE_LENGTH - FINGER_INSERTION
BASE_GROUND_Z = 0.0


@dataclass
class Keyframe:
    name: str
    duration: float
    base: np.ndarray
    active: np.ndarray
    finger_opening: float = 0.0
    tube_state: str = "source"  # source, attached, placed
    machine_1_press_z: float = PRESS_RAISED_Z
    machine_2_press_z: float = PRESS_RAISED_Z


BIN_SHIFTED_FRAME_NAMES = {
    "arrive_at_bin",
    "pregrasp",
    "insert_both_fingers",
    "wedge_and_attach_both",
    "vertical_extract_06",
    "vertical_extract_14",
    "vertical_extract_22",
    "vertical_extract_clear",
    "retreat_from_raw_rack",
    "move_to_finished_plane",
    "align_above_finished_bin",
    "descend_into_finished_holes",
    "place_finished_tubes",
    "release_finished_tubes",
    "withdraw_from_finished_bin",
    "inspection_hold",
}


def clone_frames_for_second_bin(frames: list[Keyframe]) -> list[Keyframe]:
    """Reuse identical relative IK while shifting only bin-side base poses."""
    cloned = []
    for frame in frames:
        base = frame.base.copy()
        if frame.name in BIN_SHIFTED_FRAME_NAMES:
            base[1] += BIN_CENTER_PITCH_Y
        cloned.append(replace(frame, base=base, active=frame.active.copy()))
    return cloned


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def interpolate_angle(start: float, end: float, amount: float) -> float:
    delta = (end - start + math.pi) % (2.0 * math.pi) - math.pi
    return start + amount * delta


def interpolate_base(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    result = (1.0 - amount) * start + amount * end
    result[2] = interpolate_angle(start[2], end[2], amount)
    return result


def solve_keyframes(
    tube_open_end=None,
    finished_open_end=None,
    pickup_base=None,
    finished_base=None,
    shared_solutions=None,
    shared_diagnostics=None,
):
    from single_robot_loading_kinematics import (
        BIMANUAL_ACTIVE_JOINTS,
        RobotKinematics,
        default_bimanual_seed,
    )

    kin = RobotKinematics()
    tube_open_end = TUBE_OPEN_END if tube_open_end is None else tube_open_end
    finished_open_end = (
        FINISHED_TUBE_OPEN_END if finished_open_end is None else finished_open_end
    )
    pickup_base = PICKUP_BASE if pickup_base is None else np.asarray(pickup_base)
    finished_base = (
        FINISHED_BIN_BASE if finished_base is None else np.asarray(finished_base)
    )
    # After extraction, leave the raw rack and approach machine 1 with two
    # axis-aligned translations.  Keeping yaw fixed during both translations
    # prevents the extended right arm from sweeping diagonally through the
    # machine enclosure.  The final heading change happens in place.
    raw_retreat_base = np.array(
        [MACHINE_BASE[0], pickup_base[1], pickup_base[2]], dtype=float
    )
    raw_machine_side_base = np.array(
        [MACHINE_BASE[0], MACHINE_BASE[1], pickup_base[2]], dtype=float
    )
    stow = default_bimanual_seed(kin)
    down = np.array([0.0, 0.0, -1.0])
    into_machine = np.array([-1.0, 0.0, 0.0])

    def pickup_positions(dz: float):
        return (
            tube_open_end["left"] + [0.0, 0.0, dz],
            tube_open_end["right"] + [0.0, 0.0, dz],
        )

    def machine_positions(x: float, z: float):
        return (
            np.array([x, MACHINE_HOLE_WORLD_Y["left"], z]),
            np.array([x, MACHINE_HOLE_WORLD_Y["right"], z]),
        )

    def machine_2_positions(y: float, z: float):
        return (
            np.array([MACHINE_2_HOLE_WORLD_X["left"], y, z]),
            np.array([MACHINE_2_HOLE_WORLD_X["right"], y, z]),
        )

    def finished_bin_positions(dz: float):
        return (
            finished_open_end["left"] + [0.0, 0.0, dz],
            finished_open_end["right"] + [0.0, 0.0, dz],
        )

    solutions: dict[str, np.ndarray] = dict(shared_solutions or {})
    diagnostics: dict[str, dict[str, float | bool]] = dict(shared_diagnostics or {})
    lower, upper = kin.active_bounds(BIMANUAL_ACTIVE_JOINTS)

    def solve(name, base, positions, axis, seed):
        left_position, right_position = positions
        solution, info = kin.solve_bimanual_fingers(
            base, left_position, axis, right_position, axis, seed, reference=seed
        )
        max_position_error = max(info["left_position_error"], info["right_position_error"])
        max_axis_error = max(info["left_axis_error_deg"], info["right_axis_error_deg"])
        if not info["success"] or max_position_error > 0.004 or max_axis_error > 1.0:
            raise RuntimeError(f"IK failed at {name}: {info}")
        solutions[name] = solution
        diagnostics[name] = info
        return solution

    # Start from the fully extracted posture, then continue the same IK branch
    # downward into the bin.  Reversing this list for playback gives a nearly
    # Cartesian lift; direct interpolation between unrelated endpoint basins
    # would tilt the tubes while they were still inside the holes.
    pickup_seed = np.clip(
        stow + np.random.default_rng(0).normal(0.0, 0.45, len(stow)),
        lower + 0.04,
        upper - 0.04,
    )
    pickup_seed = solve("lift_30", pickup_base, pickup_positions(0.300), down, pickup_seed)
    for name, height in (("lift_22", 0.220), ("lift_14", 0.140), ("lift_06", 0.060)):
        pickup_seed = solve(name, pickup_base, pickup_positions(height), down, pickup_seed)
    pickup_seed = solve(
        "finger_insert",
        pickup_base,
        pickup_positions(-FINGER_INSERTION),
        down,
        pickup_seed,
    )
    solve("pregrasp", pickup_base, pickup_positions(0.100), down, pickup_seed)

    if shared_solutions is None:
        machine_seed = np.clip(
            stow + np.random.default_rng(0).normal(0.0, 0.35, len(stow)),
            lower + 0.04,
            upper - 0.04,
        )
        for name, x, z in (
        # The press mesh reaches world Z~=1.817 m.  Rotate above it, then
        # descend while the hands and tubes are still outside its front face.
        ("reorient_high", -6.300, 1.950),
        ("outside_at_hole_height", -6.300, MACHINE_HOLE_WORLD_Z),
        ("preinsert", -6.616, MACHINE_HOLE_WORLD_Z),
        # Tube start = finger center - 0.434 m.  Keep the leading end at the
        # previously accepted world X=-7.500 m insertion depth.
        ("inserted", -7.066, MACHINE_HOLE_WORLD_Z),
        ("retreat", -6.300, MACHINE_HOLE_WORLD_Z),
        ):
            machine_seed = solve(name, MACHINE_BASE, machine_positions(x, z), into_machine, machine_seed)

        into_machine_2 = np.array([0.0, -1.0, 0.0])
        machine_2_seed = np.clip(
            stow + np.random.default_rng(1).normal(0.0, 0.35, len(stow)),
            lower + 0.04,
            upper - 0.04,
        )
        for name, y, z in (
        ("machine_2_high", -9.250, 1.950),
        ("machine_2_outside", -9.250, MACHINE_HOLE_WORLD_Z),
        ("machine_2_preinsert", -9.466, MACHINE_HOLE_WORLD_Z),
        # Machine 2's rotated wrist geometry extends farther along -Y than at
        # machine 1.  Stop 70 mm earlier so the complete right wrist remains
        # outside the machine face; tube start is still deep at world Y=-10.280 m.
        ("machine_2_inserted", -9.846, MACHINE_HOLE_WORLD_Z),
        ("machine_2_retreat", -9.250, MACHINE_HOLE_WORLD_Z),
        ):
            machine_2_seed = solve(
                name,
                MACHINE_2_BASE,
                machine_2_positions(y, z),
                into_machine_2,
                machine_2_seed,
            )

        machine_2_vertical_seed = np.clip(
            stow + np.random.default_rng(3).normal(0.0, 0.40, len(stow)),
            lower + 0.04,
            upper - 0.04,
        )
        machine_2_vertical_seed = solve(
            "machine_2_vertical_carry",
            MACHINE_2_BASE,
            (
                np.array([-6.000, -8.900, 1.450]),
                np.array([-6.430, -8.900, 1.450]),
            ),
            down,
            machine_2_vertical_seed,
        )

    # Finished-bin targets change with every raw-hole pair, so these three
    # solutions are refreshed even when all machine-side solutions are reused.
    finished_seed = np.clip(
        stow + np.random.default_rng(0).normal(0.0, 0.45, len(stow)),
        lower + 0.04,
        upper - 0.04,
    )
    for name, dz in (
        ("finished_above", 0.300),
        ("finished_preinsert", 0.100),
        ("finished_inserted", -FINGER_INSERTION),
    ):
        finished_seed = solve(
            name,
            finished_base,
            finished_bin_positions(dz),
            down,
            finished_seed,
        )

    frames = [
        Keyframe("home", 1.0, HOME_BASE, stow),
        Keyframe("clear_machine_corner", 3.0, np.array([-6.150, -7.650, PICKUP_BASE[2]]), stow),
        Keyframe("arrive_at_bin", 2.5, pickup_base, stow),
        Keyframe("pregrasp", 2.0, pickup_base, solutions["pregrasp"]),
        Keyframe("insert_both_fingers", 1.5, pickup_base, solutions["finger_insert"]),
        Keyframe("wedge_and_attach_both", 0.8, pickup_base, solutions["finger_insert"], 0.004, "attached"),
        Keyframe("vertical_extract_06", 0.8, pickup_base, solutions["lift_06"], 0.004, "attached"),
        Keyframe("vertical_extract_14", 0.8, pickup_base, solutions["lift_14"], 0.004, "attached"),
        Keyframe("vertical_extract_22", 0.8, pickup_base, solutions["lift_22"], 0.004, "attached"),
        Keyframe("vertical_extract_clear", 0.8, pickup_base, solutions["lift_30"], 0.004, "attached"),
        Keyframe(
            "retreat_from_raw_rack",
            2.5,
            raw_retreat_base,
            solutions["lift_30"],
            0.004,
            "attached",
        ),
        Keyframe(
            "move_sideways_to_machine_front",
            3.0,
            raw_machine_side_base,
            solutions["lift_30"],
            0.004,
            "attached",
        ),
        Keyframe(
            "turn_in_place_at_machine_1",
            1.5,
            MACHINE_BASE,
            solutions["lift_30"],
            0.004,
            "attached",
        ),
        Keyframe("machine_1_raise_before_loading", 1.5, MACHINE_BASE, solutions["lift_30"], 0.004, "attached"),
        Keyframe("rotate_both_tubes_high", 2.5, MACHINE_BASE, solutions["reorient_high"], 0.004, "attached"),
        Keyframe("descend_outside_machine", 2.0, MACHINE_BASE, solutions["outside_at_hole_height"], 0.004, "attached"),
        Keyframe("align_with_two_fixture_holes", 2.0, MACHINE_BASE, solutions["preinsert"], 0.004, "attached"),
        Keyframe("insert_both_tubes", 2.0, MACHINE_BASE, solutions["inserted"], 0.004, "attached"),
        Keyframe("release_both_tubes", 1.0, MACHINE_BASE, solutions["inserted"], 0.0, "placed"),
        Keyframe("withdraw_both_fingers", 2.0, MACHINE_BASE, solutions["retreat"], 0.0, "placed"),
        Keyframe(
            "machine_1_press_down", 1.5, MACHINE_BASE, solutions["retreat"], 0.0, "placed",
            machine_1_press_z=PRESS_DOWN_Z,
        ),
        Keyframe(
            "machine_1_press_hold", 0.8, MACHINE_BASE, solutions["retreat"], 0.0, "placed",
            machine_1_press_z=PRESS_DOWN_Z,
        ),
        Keyframe("machine_1_press_up", 1.5, MACHINE_BASE, solutions["retreat"], 0.0, "placed"),
        Keyframe("approach_machine_1_output", 1.5, MACHINE_BASE, solutions["inserted"], 0.0, "placed"),
        Keyframe("regrip_machine_1_output", 0.8, MACHINE_BASE, solutions["inserted"], 0.004, "attached"),
        Keyframe("extract_machine_1_output", 2.0, MACHINE_BASE, solutions["outside_at_hole_height"], 0.004, "attached"),
        Keyframe("raise_clear_of_machine_1", 2.0, MACHINE_BASE, solutions["reorient_high"], 0.004, "attached"),
        Keyframe("transfer_to_machine_2", 3.5, MACHINE_2_BASE, solutions["reorient_high"], 0.004, "attached"),
        Keyframe("machine_2_raise_before_loading", 1.5, MACHINE_2_BASE, solutions["reorient_high"], 0.004, "attached"),
        Keyframe("orient_above_machine_2", 2.5, MACHINE_2_BASE, solutions["machine_2_high"], 0.004, "attached"),
        Keyframe("descend_outside_machine_2", 2.0, MACHINE_2_BASE, solutions["machine_2_outside"], 0.004, "attached"),
        Keyframe("align_machine_2_fixture", 1.5, MACHINE_2_BASE, solutions["machine_2_preinsert"], 0.004, "attached"),
        Keyframe("insert_machine_2", 2.0, MACHINE_2_BASE, solutions["machine_2_inserted"], 0.004, "attached"),
        Keyframe("release_machine_2", 0.8, MACHINE_2_BASE, solutions["machine_2_inserted"], 0.0, "placed"),
        Keyframe("withdraw_from_machine_2", 1.5, MACHINE_2_BASE, solutions["machine_2_retreat"], 0.0, "placed"),
        Keyframe(
            "machine_2_press_down", 1.5, MACHINE_2_BASE, solutions["machine_2_retreat"], 0.0, "placed",
            machine_2_press_z=PRESS_DOWN_Z,
        ),
        Keyframe(
            "machine_2_press_hold", 0.8, MACHINE_2_BASE, solutions["machine_2_retreat"], 0.0, "placed",
            machine_2_press_z=PRESS_DOWN_Z,
        ),
        Keyframe("machine_2_press_up", 1.5, MACHINE_2_BASE, solutions["machine_2_retreat"], 0.0, "placed"),
        Keyframe("approach_machine_2_output", 1.5, MACHINE_2_BASE, solutions["machine_2_inserted"], 0.0, "placed"),
        Keyframe("regrip_machine_2_output", 0.8, MACHINE_2_BASE, solutions["machine_2_inserted"], 0.004, "attached"),
        Keyframe("extract_machine_2_output", 2.0, MACHINE_2_BASE, solutions["machine_2_outside"], 0.004, "attached"),
        Keyframe("restore_vertical_after_machine_2", 2.5, MACHINE_2_BASE, solutions["machine_2_vertical_carry"], 0.004, "attached"),
        Keyframe("move_vertical_clear_of_machine_2", 2.5, FINISHED_ROUTE_CLEAR_BASE, solutions["machine_2_vertical_carry"], 0.004, "attached"),
        Keyframe("move_to_finished_plane", 3.0, finished_base, solutions["machine_2_vertical_carry"], 0.004, "attached"),
        Keyframe("align_above_finished_bin", 2.0, finished_base, solutions["finished_above"], 0.004, "attached"),
        Keyframe("descend_into_finished_holes", 2.0, finished_base, solutions["finished_preinsert"], 0.004, "attached"),
        Keyframe("place_finished_tubes", 1.5, finished_base, solutions["finished_inserted"], 0.004, "attached"),
        Keyframe("release_finished_tubes", 0.8, finished_base, solutions["finished_inserted"], 0.0, "placed"),
        Keyframe("withdraw_from_finished_bin", 2.0, finished_base, solutions["finished_above"], 0.0, "placed"),
        Keyframe("inspection_hold", 2.0, finished_base, solutions["finished_above"], 0.0, "placed"),
        Keyframe("leave_finished_bin", 2.5, FINISHED_ROUTE_CLEAR_BASE, solutions["finished_above"], 0.0, "placed"),
        Keyframe("return_to_cell_center", 3.5, HOME_BASE, stow, 0.0, "placed"),
        Keyframe(
            "reset_presses_for_next_cycle", 1.5, HOME_BASE, stow, 0.0, "placed",
            machine_1_press_z=PRESS_DOWN_Z,
            machine_2_press_z=PRESS_DOWN_Z,
        ),
    ]
    machine_1_raise_index = next(
        index for index, frame in enumerate(frames) if frame.name == "machine_1_raise_before_loading"
    )
    machine_2_raise_index = next(
        index for index, frame in enumerate(frames) if frame.name == "machine_2_raise_before_loading"
    )
    for frame in frames[:machine_1_raise_index]:
        frame.machine_1_press_z = PRESS_DOWN_Z
    for frame in frames[:machine_2_raise_index]:
        frame.machine_2_press_z = PRESS_DOWN_Z
    return kin, BIMANUAL_ACTIVE_JOINTS, frames, diagnostics, solutions


def finger_and_tube_transform(kin, active_names, base: np.ndarray, active: np.ndarray, side: str):
    joint_map = dict(zip(active_names, active))
    finger = kin.fk(f"zarm_{side[0]}7_finger_link", base, joint_map)
    finger_center = (finger @ np.array([0.008307, -0.0015, 0.068487, 1.0]))[:3]
    tube_axis = -finger[:3, 2]
    tube_axis /= np.linalg.norm(tube_axis)
    tube_x = finger[:3, 0] - tube_axis * np.dot(finger[:3, 0], tube_axis)
    tube_x /= np.linalg.norm(tube_x)
    tube_z = np.cross(tube_x, tube_axis)
    tube_z /= np.linalg.norm(tube_z)
    rotation = np.column_stack((tube_x, tube_axis, tube_z))
    tube_start = finger_center - tube_axis * FINGER_TO_TUBE_START
    return finger_center, tube_start, rotation


def full_joint_vector(dof_names: list[str], active_names: list[str], active: np.ndarray, finger: float):
    values = np.zeros(len(dof_names), dtype=np.float32)
    by_name = dict(zip(active_names, active))
    by_name["zarm_l7_finger_joint"] = finger
    by_name["zarm_r7_finger_joint"] = finger
    for index, name in enumerate(dof_names):
        values[index] = by_name.get(name, 0.0)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without a viewport.")
    parser.add_argument("--check", action="store_true", help="Touch each keyframe once, then exit.")
    parser.add_argument("--once", action="store_true", help="Play once and hold the final pose.")
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Missing factory scene: {SCENE_PATH}")
    if not TUBE_ASSET_PATH.is_file():
        raise FileNotFoundError(f"Missing tube asset: {TUBE_ASSET_PATH}")

    first_spec = CONTINUOUS_PICKUP_SCHEDULE[0]
    kin, active_names, frames, diagnostics, shared_solutions = solve_keyframes(
        first_spec["open_ends"],
        first_spec["finished_open_ends"],
        pickup_base=first_spec["pickup_base"],
        finished_base=first_spec["finished_base"],
    )
    first_bin_schedule = [
        spec for spec in CONTINUOUS_PICKUP_SCHEDULE if spec["bin_index"] == 1
    ]
    cycle_frames = [frames]
    for spec in first_bin_schedule[1:]:
        _, _, solved_frames, _, _ = solve_keyframes(
            spec["open_ends"],
            spec["finished_open_ends"],
            pickup_base=spec["pickup_base"],
            finished_base=spec["finished_base"],
            shared_solutions=shared_solutions,
            shared_diagnostics=diagnostics,
        )
        cycle_frames.append(solved_frames)
    # Bin 02 has the same hole geometry and robot-relative poses as Bin 01;
    # only its world Y position differs by the 280 mm center pitch.  Reusing
    # the first-bin IK prevents the second bin from doubling startup time.
    cycle_frames.extend(clone_frames_for_second_bin(value) for value in cycle_frames[:])
    print("Bimanual two-tube loading keyframes:", flush=True)
    for name, info in diagnostics.items():
        print(
            f"  {name:16s} position_error(L/R)="
            f"{info['left_position_error'] * 1000.0:.3f}/"
            f"{info['right_position_error'] * 1000.0:.3f} mm "
            f"axis_error(L/R)={info['left_axis_error_deg']:.4f}/"
            f"{info['right_axis_error_deg']:.4f} deg",
            flush=True,
        )
    print(
        "Continuous pickup schedule: "
        + " -> ".join(spec["label"] for spec in CONTINUOUS_PICKUP_SCHEDULE),
        flush=True,
    )

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    app = SimulationApp({"headless": args.headless})
    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import is_stage_loading, open_stage
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        if not open_stage(str(SCENE_PATH)):
            raise RuntimeError(f"Could not open {SCENE_PATH}")
        while is_stage_loading():
            app.update()
        for _ in range(5):
            app.update()

        stage = omni.usd.get_context().get_stage()
        if stage is None or not stage.GetPrimAtPath(ROBOT_PATH):
            raise RuntimeError(f"Missing robot articulation: {ROBOT_PATH}")

        # Only one robot participates in this debug session.  Removing the
        # other eleven live instances avoids unnecessary physics work and does
        # not modify the factory USD on disk.
        robots_root = stage.GetPrimAtPath("/World/Factory/Robots")
        for station in list(robots_root.GetChildren()):
            if str(station.GetPath()) != ROBOT_STATION_PATH:
                stage.RemovePrim(station.GetPath())

        # The first-stage motion preview is explicitly kinematic.  Disable the
        # robot collision shapes in this unsaved session so the floor cannot
        # inject a contact impulse after we place the chassis at Z=0.  Visual
        # clearance is still checked by the authored keyframe route.
        disabled_collision_count = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                disabled_collision_count += 1

        source_image_sets = []
        for spec in CONTINUOUS_PICKUP_SCHEDULE:
            image_set = {}
            for side in ("left", "right"):
                source_tube = stage.GetPrimAtPath(spec["paths"][side])
                if not source_tube:
                    raise RuntimeError(f"Missing {side} source tube: {spec['paths'][side]}")
                image_set[side] = UsdGeom.Imageable(source_tube)
            source_image_sets.append(image_set)
        source_images = source_image_sets[0]
        carried_image_sets = []
        translate_op_sets = []
        orient_op_sets = []
        for cycle_number in range(1, len(CONTINUOUS_PICKUP_SCHEDULE) + 1):
            image_set = {}
            translate_set = {}
            orient_set = {}
            for side in ("left", "right"):
                carried_path = (
                    f"/World/SingleRobotLoadingDemo/Cycle_{cycle_number:02d}/"
                    f"Tube_{side.capitalize()}"
                )
                carried_xform = UsdGeom.Xform.Define(stage, carried_path)
                carried_xform.GetPrim().GetReferences().AddReference(str(TUBE_ASSET_PATH))
                image_set[side] = UsdGeom.Imageable(carried_xform.GetPrim())
                image_set[side].MakeInvisible()
                translate_set[side] = carried_xform.AddTranslateOp(
                    UsdGeom.XformOp.PrecisionDouble
                )
                orient_set[side] = carried_xform.AddOrientOp(
                    UsdGeom.XformOp.PrecisionDouble
                )
            carried_image_sets.append(image_set)
            translate_op_sets.append(translate_set)
            orient_op_sets.append(orient_set)
        carried_images = carried_image_sets[0]
        translate_ops = translate_op_sets[0]
        orient_ops = orient_op_sets[0]

        press_ops = {}
        for name, path in (
            ("machine_1", MACHINE_1_PRESS_PATH),
            ("machine_2", MACHINE_2_PRESS_PATH),
        ):
            press_prim = stage.GetPrimAtPath(path)
            if not press_prim:
                raise RuntimeError(f"Missing press mesh: {path}")
            press_ops[name] = UsdGeom.Xformable(press_prim).AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble,
                opSuffix="loadingCycle",
            )

        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / args.fps, rendering_dt=1.0 / args.fps)
        robot = world.scene.add(SingleArticulation(prim_path=ROBOT_PATH, name="loading_robot"))
        world.reset()
        if not robot.handles_initialized:
            raise RuntimeError("Robot articulation did not initialize")
        robot.disable_gravity()
        missing = sorted(
            set(active_names + ["zarm_l7_finger_joint", "zarm_r7_finger_joint"])
            - set(robot.dof_names)
        )
        if missing:
            raise RuntimeError(f"Robot USD is missing required DOFs: {missing}")
        print(f"Articulation ready: {robot.num_dof} DOFs", flush=True)

        if not args.headless:
            try:
                from isaacsim.core.utils.viewports import set_camera_view

                set_camera_view(
                    eye=np.array([-3.60, -6.40, 3.10]),
                    target=np.array([-6.35, -9.55, 1.00]),
                    camera_prim_path="/World/Camera",
                )
            except Exception as exc:
                print(f"Debug camera setup skipped: {exc}", flush=True)

        placed_transforms = {"left": None, "right": None}
        last_state = None

        def apply_pose(
            base,
            active,
            finger_opening,
            tube_state,
            machine_1_press_z,
            machine_2_press_z,
        ):
            nonlocal placed_transforms, last_state
            quaternion = np.array(
                [math.cos(base[2] / 2.0), 0.0, 0.0, math.sin(base[2] / 2.0)],
                dtype=np.float32,
            )
            # This preview is kinematic: the wheeled chassis must remain in
            # contact with the flat floor.  Re-author the exact ground height
            # and clear all root velocity on every rendered frame so PhysX
            # cannot add a vertical impulse and make the base appear to hop.
            robot.set_world_pose(
                position=np.array([base[0], base[1], BASE_GROUND_Z]),
                orientation=quaternion,
            )
            robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
            robot.set_joint_positions(full_joint_vector(robot.dof_names, active_names, active, finger_opening))
            robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))
            press_ops["machine_1"].Set(Gf.Vec3d(0.0, 0.0, machine_1_press_z))
            press_ops["machine_2"].Set(Gf.Vec3d(0.0, 0.0, machine_2_press_z))

            for side in ("left", "right"):
                _, tube_start, tube_rotation = finger_and_tube_transform(
                    kin, active_names, base, active, side
                )
                if tube_state == "source":
                    source_images[side].MakeVisible()
                    carried_images[side].MakeInvisible()
                else:
                    source_images[side].MakeInvisible()
                    carried_images[side].MakeVisible()
                    if tube_state == "attached":
                        placed_transforms[side] = None
                        current_transform = (tube_start.copy(), tube_rotation.copy())
                    else:
                        if placed_transforms[side] is None:
                            placed_transforms[side] = (tube_start.copy(), tube_rotation.copy())
                        current_transform = placed_transforms[side]
                    position, matrix = current_transform
                    quat_xyzw = Rotation.from_matrix(matrix).as_quat()
                    translate_ops[side].Set(Gf.Vec3d(*position))
                    orient_ops[side].Set(Gf.Quatd(quat_xyzw[3], Gf.Vec3d(*quat_xyzw[:3])))
            if tube_state != last_state:
                print(f"Tube state: {tube_state}", flush=True)
                last_state = tube_state

        if args.check:
            for cycle_index, frames in enumerate(cycle_frames, start=1):
                source_images = source_image_sets[cycle_index - 1]
                carried_images = carried_image_sets[cycle_index - 1]
                translate_ops = translate_op_sets[cycle_index - 1]
                orient_ops = orient_op_sets[cycle_index - 1]
                for frame in frames:
                    apply_pose(
                        frame.base.copy(), frame.active.copy(), frame.finger_opening, frame.tube_state,
                        frame.machine_1_press_z, frame.machine_2_press_z,
                    )
                    world.step(render=not args.headless)
                print(
                    f"CHECK CYCLE {cycle_index}: {CONTINUOUS_PICKUP_SCHEDULE[cycle_index - 1]['label']}",
                    flush=True,
                )
            print("CHECK PASSED: all continuous-cycle keyframes applied", flush=True)
            return

        print("Playing: raw bin -> press 1 -> press 2 -> finished bin", flush=True)
        print(f"Chassis ground lock: world Z={BASE_GROUND_Z:.3f} m", flush=True)
        print(
            f"Kinematic preview: gravity disabled, {disabled_collision_count} robot collision shapes disabled",
            flush=True,
        )
        cycle_index = 0
        while app.is_running():
            frames = cycle_frames[cycle_index]
            source_images = source_image_sets[cycle_index]
            carried_images = carried_image_sets[cycle_index]
            translate_ops = translate_op_sets[cycle_index]
            orient_ops = orient_op_sets[cycle_index]
            print(
                f"Cycle {cycle_index + 1}/{len(cycle_frames)} pickup "
                f"{CONTINUOUS_PICKUP_SCHEDULE[cycle_index]['label']}",
                flush=True,
            )
            placed_transforms = {"left": None, "right": None}
            last_state = None
            apply_pose(
                frames[0].base.copy(), frames[0].active.copy(), frames[0].finger_opening,
                frames[0].tube_state, frames[0].machine_1_press_z, frames[0].machine_2_press_z,
            )
            world.step(render=True)
            for previous, current in zip(frames, frames[1:]):
                print(f"Phase: {current.name}", flush=True)
                count = max(2, int(round(current.duration * args.fps)))
                for index in range(1, count + 1):
                    if not app.is_running():
                        return
                    amount = smoothstep(index / count)
                    base = interpolate_base(previous.base, current.base, amount)
                    active = (1.0 - amount) * previous.active + amount * current.active
                    finger = (1.0 - amount) * previous.finger_opening + amount * current.finger_opening
                    machine_1_press_z = (
                        (1.0 - amount) * previous.machine_1_press_z
                        + amount * current.machine_1_press_z
                    )
                    machine_2_press_z = (
                        (1.0 - amount) * previous.machine_2_press_z
                        + amount * current.machine_2_press_z
                    )
                    apply_pose(
                        base, active, finger, current.tube_state,
                        machine_1_press_z, machine_2_press_z,
                    )
                    world.step(render=True)
            if args.once:
                while app.is_running():
                    apply_pose(
                        frames[-1].base, frames[-1].active, frames[-1].finger_opening, "placed",
                        frames[-1].machine_1_press_z, frames[-1].machine_2_press_z,
                    )
                    world.step(render=True)
                return
            cycle_index += 1
            if cycle_index == len(cycle_frames):
                print(
                    "BATCH COMPLETE: two raw bins empty, two finished bins contain 96 tubes",
                    flush=True,
                )
                while app.is_running():
                    apply_pose(
                        frames[-1].base,
                        frames[-1].active,
                        frames[-1].finger_opening,
                        "placed",
                        frames[-1].machine_1_press_z,
                        frames[-1].machine_2_press_z,
                    )
                    world.step(render=True)
                return
    finally:
        app.close()


if __name__ == "__main__":
    main()
