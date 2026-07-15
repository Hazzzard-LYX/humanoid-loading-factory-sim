#!/usr/bin/env python3
"""Run the accepted two-tube loading cycle in all twelve factory cells."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from create_factory_scene import robot_workpoints
from preview_single_robot_loading import (
    BASE_GROUND_Z,
    CONTINUOUS_PICKUP_SCHEDULE,
    ROOT_DIR,
    SCENE_PATH,
    TUBE_ASSET_PATH,
    clone_frames_for_second_bin,
    finger_and_tube_transform,
    full_joint_vector,
    interpolate_base,
    solve_keyframes,
    smoothstep,
)


REFERENCE_Y = -9.065
REFERENCE_DEMO_ROOT = "/World/AllCellsLoadingDemo"
MIRROR_WORLD_X = np.diag([-1.0, 1.0, 1.0])
# A second reflection in tube-local X keeps the transformed orientation proper
# while preserving the tube's longitudinal local-Y axis.
MIRROR_TUBE_LOCAL_X = np.diag([-1.0, 1.0, 1.0])
ARM_MIRROR_SIGNS = (1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0)


@dataclass(frozen=True)
class CellSpec:
    index: int
    bank: str
    pair: int
    east: bool
    delta_y: float
    station_path: str
    robot_path: str
    machine_1_press_path: str
    machine_2_press_path: str
    source_tubes_root: str

    @property
    def key(self) -> str:
        side = "East" if self.east else "West"
        return f"Cell_{self.index:02d}_{side}_Pair_{self.pair:02d}"


@dataclass
class CellRuntime:
    spec: CellSpec
    robot: object
    press_ops: dict
    source_image_sets: list
    carried_image_sets: list
    translate_op_sets: list
    orient_op_sets: list
    placed_transforms: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run one two-tube cycle and hold.")
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def build_cell_specs() -> list[CellSpec]:
    specs = []
    for index, (bank, pair, _x, y, _yaw) in enumerate(robot_workpoints(), start=1):
        east = bank == "Bank_02_East"
        side = "East" if east else "West"
        station_path = f"/World/Factory/Robots/Robot_{index:02d}_{side}_Pair_{pair:02d}"
        cell_root = f"/World/Factory/Machines/{bank}/Cell_{pair:02d}"
        rack_root = f"/World/Factory/MaterialRacks/{bank}/Cell_{pair:02d}"
        specs.append(
            CellSpec(
                index=index,
                bank=bank,
                pair=pair,
                east=east,
                delta_y=y - REFERENCE_Y,
                station_path=station_path,
                robot_path=f"{station_path}/Model",
                machine_1_press_path=(
                    f"{cell_root}/Machine_01_Wall_Process01/Press"
                ),
                machine_2_press_path=(
                    f"{cell_root}/Machine_02_Aisle_Process02/Press"
                ),
                source_tubes_root=(
                    f"{rack_root}/RawMaterialRack_Machine01_MirroredSide/"
                    "Bins/Bin_01/WorldYawAlignment/AisleAxisTilt/Orientation/Tubes"
                ),
            )
        )
    return specs


def transform_base(base: np.ndarray, spec: CellSpec) -> np.ndarray:
    result = np.asarray(base, dtype=float).copy()
    result[1] += spec.delta_y
    if spec.east:
        result[0] = -result[0]
        result[2] = math.pi - result[2]
    result[2] = (result[2] + math.pi) % (2.0 * math.pi) - math.pi
    return result


def transform_tube_pose(position: np.ndarray, matrix: np.ndarray, spec: CellSpec):
    transformed_position = np.asarray(position, dtype=float).copy()
    transformed_position[1] += spec.delta_y
    transformed_matrix = np.asarray(matrix, dtype=float).copy()
    if spec.east:
        transformed_position[0] = -transformed_position[0]
        transformed_matrix = MIRROR_WORLD_X @ transformed_matrix @ MIRROR_TUBE_LOCAL_X
    return transformed_position, transformed_matrix


def mirror_active(active_names: list[str], active: np.ndarray) -> np.ndarray:
    """Mirror a west-cell pose into the symmetric east-cell robot."""
    source = dict(zip(active_names, np.asarray(active, dtype=float)))
    mirrored = dict(source)
    mirrored["waist_yaw_joint"] = -source["waist_yaw_joint"]
    for index, sign in enumerate(ARM_MIRROR_SIGNS, start=1):
        mirrored[f"zarm_l{index}_joint"] = sign * source[f"zarm_r{index}_joint"]
        mirrored[f"zarm_r{index}_joint"] = sign * source[f"zarm_l{index}_joint"]
    return np.array([mirrored[name] for name in active_names], dtype=float)


def solve_all_cycles():
    first = CONTINUOUS_PICKUP_SCHEDULE[0]
    kin, active_names, frames, diagnostics, shared_solutions = solve_keyframes(
        first["open_ends"],
        first["finished_open_ends"],
        pickup_base=first["pickup_base"],
        finished_base=first["finished_base"],
    )
    first_bin_schedule = [
        spec for spec in CONTINUOUS_PICKUP_SCHEDULE if spec["bin_index"] == 1
    ]
    cycle_frames = [frames]
    for spec in first_bin_schedule[1:]:
        _, _, frames, _, _ = solve_keyframes(
            spec["open_ends"],
            spec["finished_open_ends"],
            pickup_base=spec["pickup_base"],
            finished_base=spec["finished_base"],
            shared_solutions=shared_solutions,
            shared_diagnostics=diagnostics,
        )
        cycle_frames.append(frames)
    cycle_frames.extend(clone_frames_for_second_bin(value) for value in cycle_frames[:])
    return kin, active_names, cycle_frames


def main() -> None:
    args = parse_args()
    if not SCENE_PATH.is_file() or not TUBE_ASSET_PATH.is_file():
        raise FileNotFoundError("Factory scene or aluminum-tube asset is missing")

    cell_specs = build_cell_specs()
    if len(cell_specs) != 12:
        raise RuntimeError(f"Expected 12 cells, found {len(cell_specs)}")
    kin, active_names, cycle_frames = solve_all_cycles()
    print(
        f"Prepared {len(cycle_frames)} cycles for {len(cell_specs)} cells "
        f"({len(cell_specs) * len(cycle_frames) * 2} finished tubes total)",
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

        disabled_collision_count = 0
        for spec in cell_specs:
            robot_prim = stage.GetPrimAtPath(spec.robot_path)
            if not robot_prim:
                raise RuntimeError(f"Missing robot: {spec.robot_path}")
            for prim in Usd.PrimRange(robot_prim):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                    disabled_collision_count += 1

        runtime_builders = []
        for spec in cell_specs:
            source_image_sets = []
            carried_image_sets = []
            translate_op_sets = []
            orient_op_sets = []
            for cycle_number, pickup in enumerate(CONTINUOUS_PICKUP_SCHEDULE, start=1):
                sources = {}
                carried = {}
                translates = {}
                orients = {}
                for side in ("left", "right"):
                    column_path = pickup["paths"][side].rsplit("/", 1)[-1]
                    source_root = spec.source_tubes_root.replace(
                        "/Bin_01/",
                        f"/Bin_{pickup['bin_index']:02d}/",
                    )
                    source_prim = stage.GetPrimAtPath(f"{source_root}/{column_path}")
                    if not source_prim:
                        raise RuntimeError(
                            f"Missing source tube in {spec.key}: {column_path}"
                        )
                    sources[side] = UsdGeom.Imageable(source_prim)
                    carried_path = (
                        f"{REFERENCE_DEMO_ROOT}/{spec.key}/Cycle_{cycle_number:02d}/"
                        f"Tube_{side.capitalize()}"
                    )
                    carried_xform = UsdGeom.Xform.Define(stage, carried_path)
                    carried_xform.GetPrim().GetReferences().AddReference(str(TUBE_ASSET_PATH))
                    carried[side] = UsdGeom.Imageable(carried_xform.GetPrim())
                    carried[side].MakeInvisible()
                    translates[side] = carried_xform.AddTranslateOp(
                        UsdGeom.XformOp.PrecisionDouble
                    )
                    orients[side] = carried_xform.AddOrientOp(
                        UsdGeom.XformOp.PrecisionDouble
                    )
                source_image_sets.append(sources)
                carried_image_sets.append(carried)
                translate_op_sets.append(translates)
                orient_op_sets.append(orients)

            press_ops = {}
            for name, path in (
                ("machine_1", spec.machine_1_press_path),
                ("machine_2", spec.machine_2_press_path),
            ):
                prim = stage.GetPrimAtPath(path)
                if not prim:
                    raise RuntimeError(f"Missing press: {path}")
                press_ops[name] = UsdGeom.Xformable(prim).AddTranslateOp(
                    UsdGeom.XformOp.PrecisionDouble,
                    opSuffix="allCellsLoadingCycle",
                )
            runtime_builders.append(
                (
                    spec,
                    press_ops,
                    source_image_sets,
                    carried_image_sets,
                    translate_op_sets,
                    orient_op_sets,
                )
            )

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / args.fps,
            rendering_dt=1.0 / args.fps,
        )
        runtimes = []
        for builder in runtime_builders:
            spec = builder[0]
            robot = world.scene.add(
                SingleArticulation(prim_path=spec.robot_path, name=f"loading_robot_{spec.index:02d}")
            )
            runtimes.append(
                CellRuntime(
                    spec=spec,
                    robot=robot,
                    press_ops=builder[1],
                    source_image_sets=builder[2],
                    carried_image_sets=builder[3],
                    translate_op_sets=builder[4],
                    orient_op_sets=builder[5],
                    placed_transforms={"left": None, "right": None},
                )
            )
        world.reset()
        for runtime in runtimes:
            if not runtime.robot.handles_initialized:
                raise RuntimeError(f"Articulation did not initialize: {runtime.spec.robot_path}")
            runtime.robot.disable_gravity()
            missing = sorted(set(active_names) - set(runtime.robot.dof_names))
            if missing:
                raise RuntimeError(f"{runtime.spec.key} missing DOFs: {missing}")

        if not args.headless:
            try:
                from isaacsim.core.utils.viewports import set_camera_view

                set_camera_view(
                    eye=np.array([20.5, -29.0, 22.0]),
                    target=np.array([0.0, 0.5, 0.8]),
                    camera_prim_path="/World/Camera",
                )
            except Exception as exc:
                print(f"Camera setup skipped: {exc}", flush=True)

        def apply_pose(cycle_index, base, active, finger, tube_state, press_1_z, press_2_z):
            reference_tubes = {}
            for side in ("left", "right"):
                _, position, matrix = finger_and_tube_transform(
                    kin, active_names, base, active, side
                )
                reference_tubes[side] = (position, matrix)

            for runtime in runtimes:
                spec = runtime.spec
                robot_base = transform_base(base, spec)
                robot_active = mirror_active(active_names, active) if spec.east else active
                quaternion = np.array(
                    [
                        math.cos(robot_base[2] / 2.0),
                        0.0,
                        0.0,
                        math.sin(robot_base[2] / 2.0),
                    ],
                    dtype=np.float32,
                )
                runtime.robot.set_world_pose(
                    position=np.array([robot_base[0], robot_base[1], BASE_GROUND_Z]),
                    orientation=quaternion,
                )
                runtime.robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
                runtime.robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
                runtime.robot.set_joint_positions(
                    full_joint_vector(
                        runtime.robot.dof_names, active_names, robot_active, finger
                    )
                )
                runtime.robot.set_joint_velocities(
                    np.zeros(runtime.robot.num_dof, dtype=np.float32)
                )
                runtime.press_ops["machine_1"].Set(Gf.Vec3d(0.0, 0.0, press_1_z))
                runtime.press_ops["machine_2"].Set(Gf.Vec3d(0.0, 0.0, press_2_z))

                sources = runtime.source_image_sets[cycle_index]
                carried = runtime.carried_image_sets[cycle_index]
                translates = runtime.translate_op_sets[cycle_index]
                orients = runtime.orient_op_sets[cycle_index]
                for side in ("left", "right"):
                    if tube_state == "source":
                        sources[side].MakeVisible()
                        carried[side].MakeInvisible()
                        continue
                    sources[side].MakeInvisible()
                    carried[side].MakeVisible()
                    current = transform_tube_pose(*reference_tubes[side], spec)
                    if tube_state == "attached":
                        runtime.placed_transforms[side] = None
                    elif runtime.placed_transforms[side] is None:
                        runtime.placed_transforms[side] = current
                    if tube_state == "placed":
                        current = runtime.placed_transforms[side]
                    position, matrix = current
                    quat_xyzw = Rotation.from_matrix(matrix).as_quat()
                    translates[side].Set(Gf.Vec3d(*position))
                    orients[side].Set(
                        Gf.Quatd(quat_xyzw[3], Gf.Vec3d(*quat_xyzw[:3]))
                    )

        if args.check:
            for cycle_index, frames in enumerate(cycle_frames):
                for runtime in runtimes:
                    runtime.placed_transforms = {"left": None, "right": None}
                for frame in frames:
                    apply_pose(
                        cycle_index,
                        frame.base,
                        frame.active,
                        frame.finger_opening,
                        frame.tube_state,
                        frame.machine_1_press_z,
                        frame.machine_2_press_z,
                    )
                    world.step(render=False)
                print(f"CHECK ALL CELLS CYCLE {cycle_index + 1}", flush=True)
            print(
                f"CHECK PASSED: all 12 cells and all {len(cycle_frames)} cycles",
                flush=True,
            )
            return

        print(
            f"Playing synchronized loading in all 12 cells; "
            f"disabled {disabled_collision_count} robot collision shapes",
            flush=True,
        )
        cycle_limit = 1 if args.once else len(cycle_frames)
        for cycle_index in range(cycle_limit):
            frames = cycle_frames[cycle_index]
            for runtime in runtimes:
                runtime.placed_transforms = {"left": None, "right": None}
            print(
                f"ALL CELLS CYCLE {cycle_index + 1}/{cycle_limit}: "
                f"{CONTINUOUS_PICKUP_SCHEDULE[cycle_index]['label']}",
                flush=True,
            )
            first = frames[0]
            apply_pose(
                cycle_index,
                first.base,
                first.active,
                first.finger_opening,
                first.tube_state,
                first.machine_1_press_z,
                first.machine_2_press_z,
            )
            world.step(render=True)
            for previous, current in zip(frames, frames[1:]):
                # Twelve articulated robots make one rendered simulation step
                # substantially more expensive than in the single-cell preview.
                # Drive interpolation from wall-clock time so a low viewport
                # frame rate drops intermediate samples instead of stretching a
                # nominal two-second motion into a much longer slow-motion move.
                phase_started = time.perf_counter()
                while True:
                    if not app.is_running():
                        return
                    linear_amount = min(
                        (time.perf_counter() - phase_started) / current.duration,
                        1.0,
                    )
                    amount = smoothstep(linear_amount)
                    apply_pose(
                        cycle_index,
                        interpolate_base(previous.base, current.base, amount),
                        (1.0 - amount) * previous.active + amount * current.active,
                        (1.0 - amount) * previous.finger_opening
                        + amount * current.finger_opening,
                        current.tube_state,
                        (1.0 - amount) * previous.machine_1_press_z
                        + amount * current.machine_1_press_z,
                        (1.0 - amount) * previous.machine_2_press_z
                        + amount * current.machine_2_press_z,
                    )
                    world.step(render=True)
                    if linear_amount >= 1.0:
                        break

        print(
            "ALL CELLS BATCH COMPLETE: 24 raw bins empty, "
            f"24 finished bins contain {12 * cycle_limit * 2} tubes",
            flush=True,
        )
        final_cycle = cycle_limit - 1
        final = cycle_frames[final_cycle][-1]
        while app.is_running():
            apply_pose(
                final_cycle,
                final.base,
                final.active,
                final.finger_opening,
                "placed",
                final.machine_1_press_z,
                final.machine_2_press_z,
            )
            world.step(render=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
