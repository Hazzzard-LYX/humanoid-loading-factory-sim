#!/usr/bin/env python3
"""Open the generated humanoid loading factory scene in Isaac Sim."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT_DIR / "scenes" / "humanoid_loading_factory.usd"


def main() -> None:
    scene_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_SCENE
    if not scene_path.is_file():
        raise FileNotFoundError(
            f"Scene not found: {scene_path}. Run scripts/create_factory_scene.py --headless first."
        )

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    app = SimulationApp({"headless": False})
    try:
        import omni.usd
        from isaacsim.core.utils.stage import is_stage_loading, open_stage

        for _ in range(5):
            app.update()
        if not open_stage(str(scene_path)):
            raise RuntimeError(f"Failed to open stage: {scene_path}")
        while is_stage_loading():
            app.update()

        stage = omni.usd.get_context().get_stage()
        default_prim = stage.GetDefaultPrim() if stage else None
        if not default_prim or not default_prim.GetChildren():
            raise RuntimeError("Loaded factory stage is empty.")

        robot_parent = "/World/Factory/Robots"
        material_rack_prefix = "/World/Factory/MaterialRacks/"
        robot_count = 0
        rack_count = 0
        bin_count = 0
        raw_tube_count = 0
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            prim_name = prim.GetName()
            if str(prim.GetParent().GetPath()) == robot_parent and prim_name.startswith("Robot_"):
                robot_count += 1
            if prim_path.startswith(material_rack_prefix):
                if prim_name.startswith(("RawMaterialRack_", "FinishedRack_")):
                    rack_count += 1
                elif prim_name.startswith("Bin_"):
                    bin_count += 1
                elif prim_name.startswith("Tube_R"):
                    raw_tube_count += 1

        expected_counts = (12, 24, 48, 1152)
        actual_counts = (robot_count, rack_count, bin_count, raw_tube_count)
        if actual_counts != expected_counts:
            raise RuntimeError(
                "Factory asset count mismatch: "
                f"expected robots/racks/bins/raw_tubes={expected_counts}, got {actual_counts}"
            )

        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.set_active_camera("/World/Camera")
        except Exception as exc:
            print(f"Viewport camera selection skipped: {exc}", flush=True)

        print(f"Opened scene: {scene_path}", flush=True)
        print(f"Default prim: {default_prim.GetPath()}", flush=True)
        print(
            "Factory: 24 lathes in two continuous 12-machine banks; central stockpile areas; "
            "two 2.5 m longitudinal aisles and one 2.5 m transverse aisle",
            flush=True,
        )
        print(
            "L cells: aisle-side machines parallel; wall-side machines face aisles; "
            "axis intersection local coordinate (0.935, 0.000) m in both frames",
            flush=True,
        )
        print(
            f"Robots: {robot_count} humanoid robots, one at each paired-machine axis intersection",
            flush=True,
        )
        print(
            f"Material handling: {rack_count} supplied STL stands and {bin_count} supplied STL bins "
            "(12 raw + 12 finished stands, two side-by-side bins per stand)",
            flush=True,
        )
        print(
            f"Raw stock: {raw_tube_count} aluminum-tube instances "
            "(48 usable outer-ring holes per raw bin; 12 inner holes blocked)",
            flush=True,
        )
        while app.is_running():
            app.update()
    finally:
        app.close()


if __name__ == "__main__":
    main()
