#!/usr/bin/env python3
"""Inspect exact world-space targets for the first west loading cell."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT_DIR / "scenes" / "humanoid_loading_factory.usd"

ROBOT_PATH = "/World/Factory/Robots/Robot_01_West_Pair_01/Model"
RAW_BIN_PATH = (
    "/World/Factory/MaterialRacks/Bank_01_West/Cell_01/"
    "RawMaterialRack_Machine01_MirroredSide/Bins/Bin_01/"
    "WorldYawAlignment/AisleAxisTilt/Orientation"
)
TUBE_PATH = f"{RAW_BIN_PATH}/Tubes/Tube_R01_C05"
MACHINE_PATH = (
    "/World/Factory/Machines/Bank_01_West/Cell_01/"
    "Machine_01_Wall_Process01"
)


def main() -> None:
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.Open(str(SCENE_PATH))
        if stage is None:
            raise RuntimeError(f"Could not open {SCENE_PATH}")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        def matrix(path: str):
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise RuntimeError(f"Missing prim: {path}")
            return cache.GetLocalToWorldTransform(prim)

        def point(path: str, xyz=(0.0, 0.0, 0.0)):
            return matrix(path).Transform(Gf.Vec3d(*xyz))

        def direction(path: str, xyz):
            value = matrix(path).TransformDir(Gf.Vec3d(*xyz))
            return value.GetNormalized()

        tube_start = point(TUBE_PATH)
        tube_axis = direction(TUBE_PATH, (0.0, 1.0, 0.0))
        tube_hole_face = tube_start + tube_axis * 0.275
        tube_open_end = tube_start + tube_axis * 0.452

        raw_tubes_root = stage.GetPrimAtPath(f"{RAW_BIN_PATH}/Tubes")
        robot_base = point(f"{ROBOT_PATH}/base_link")
        tube_candidates = []
        for tube_prim in raw_tubes_root.GetChildren():
            candidate_path = str(tube_prim.GetPath())
            candidate_start = point(candidate_path)
            candidate_axis = direction(candidate_path, (0.0, 1.0, 0.0))
            candidate_open = candidate_start + candidate_axis * 0.452
            planar_distance = (
                (candidate_open[0] - robot_base[0]) ** 2
                + (candidate_open[1] - robot_base[1]) ** 2
            ) ** 0.5
            tube_candidates.append((planar_distance, tube_prim.GetName(), candidate_open))
        tube_candidates.sort(key=lambda item: item[0])

        machine_entrance = point(MACHINE_PATH, (0.02, 0.0, 0.72))
        machine_inserted = point(MACHINE_PATH, (-0.20, 0.0, 0.72))
        machine_axis = (machine_inserted - machine_entrance).GetNormalized()

        print(f"SCENE={SCENE_PATH}", flush=True)
        print(f"ROBOT_BASE={tuple(round(v, 6) for v in point(f'{ROBOT_PATH}/base_link'))}", flush=True)
        for side in ("l", "r"):
            wrist = f"{ROBOT_PATH}/zarm_{side}7_link"
            finger = f"{ROBOT_PATH}/zarm_{side}7_finger_link"
            print(
                f"REST_{side.upper()}_WRIST={tuple(round(v, 6) for v in point(wrist))} "
                f"FINGER_CENTER={tuple(round(v, 6) for v in point(finger, (0.008307, -0.0015, 0.068487)))}",
                flush=True,
            )
        print(
            f"TUBE_START={tuple(round(v, 6) for v in tube_start)} "
            f"HOLE_FACE={tuple(round(v, 6) for v in tube_hole_face)} "
            f"OPEN_END={tuple(round(v, 6) for v in tube_open_end)} "
            f"AXIS={tuple(round(v, 6) for v in tube_axis)}",
            flush=True,
        )
        for distance_value, name, candidate_open in tube_candidates[:5]:
            print(
                f"NEAREST_TUBE={name} DIST_XY={distance_value:.6f} "
                f"OPEN_END={tuple(round(v, 6) for v in candidate_open)}",
                flush=True,
            )
        print(
            f"MACHINE_ENTRANCE={tuple(round(v, 6) for v in machine_entrance)} "
            f"INSERTED={tuple(round(v, 6) for v in machine_inserted)} "
            f"AXIS={tuple(round(v, 6) for v in machine_axis)}",
            flush=True,
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
