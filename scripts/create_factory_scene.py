#!/usr/bin/env python3
"""Build the humanoid loading factory scene for Isaac Sim.

The layout has two continuous twelve-machine banks outside two 2.5 m
longitudinal aisles. Each bank contains six evenly spaced perpendicular pairs.
The area between the longitudinal aisles is reserved for material stockpiles.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT_DIR / "biped_s62" / "工厂场景"
LATHE_MESH_DIR = ASSET_ROOT / "车床" / "meshes"
BIN_STL_PATH = ASSET_ROOT / "料箱.STL"
BIN_STAND_STL_PATH = ASSET_ROOT / "料箱座.STL"
ALUMINUM_TUBE_STL_PATH = ASSET_ROOT / "铝管.STL"
DEFAULT_OUTPUT = ROOT_DIR / "scenes" / "humanoid_loading_factory.usd"
ROBOT_URDF_PATH = ROOT_DIR / "biped_s62" / "urdf" / "biped_s62.urdf"
ROBOT_MESH_DIR = ROOT_DIR / "biped_s62" / "meshes"
ROBOT_ASSET_PATH = ROOT_DIR / "assets" / "robots" / "biped_s62" / "biped_s62.usd"
AGV_URDF_PATH = ASSET_ROOT / "HIK-Q2-400D" / "urdf" / "HIK-Q2-400D.urdf"
AGV_MESH_DIR = ASSET_ROOT / "HIK-Q2-400D" / "meshes"
AGV_ASSET_PATH = ROOT_DIR / "assets" / "agv" / "HIK-Q2-400D.usd"
ALUMINUM_TUBE_ASSET_PATH = ROOT_DIR / "assets" / "materials" / "aluminum_tube.usd"

# Measured from the supplied lathe STL files (metres).
MACHINE_SIZE_X = 1.60
MACHINE_SIZE_Y = 1.17
MACHINE_SIZE_Z = 2.09
MACHINE_LOCAL_CENTER = (-0.70, 0.0)
MACHINE_LOCAL_Z_MIN = -0.380775

MACHINE_BANK_COUNT = 2
L_PAIRS_PER_BANK = 6
MACHINES_PER_L_PAIR = 2
MACHINES_PER_BANK = L_PAIRS_PER_BANK * MACHINES_PER_L_PAIR
TOTAL_MACHINE_COUNT = MACHINE_BANK_COUNT * MACHINES_PER_BANK

LONGITUDINAL_AISLE_WIDTH = 2.5
TRANSVERSE_AISLE_WIDTH = 2.5
LONGITUDINAL_AISLE_CENTERS_X = (-3.15, 3.15)
ROOM_SIZE_X = 18.5
# Increase the longitudinal pitch between neighboring cells.  The symmetric
# room extension grows the six-cell pitch from 3.996 m to 4.796 m while
# retaining the transverse aisle at the room center.
ROOM_SIZE_Y = 29.0

WALL_MACHINE_CLEARANCE_X = 0.50
L_PAIR_MACHINE_GAP_X = 0.25
MACHINE_BANK_END_MARGIN_Y = 1.00
L_PAIR_AXIS_INTERSECTION_FROM_CENTER = (
    MACHINE_SIZE_X / 2.0 + MACHINE_SIZE_Y / 2.0 + L_PAIR_MACHINE_GAP_X
)
ROBOT_WORKPOINT_LOCAL_X = MACHINE_LOCAL_CENTER[0] + L_PAIR_AXIS_INTERSECTION_FROM_CENTER
ROBOT_WORKPOINT_LOCAL_Y = MACHINE_LOCAL_CENTER[1]
ROBOT_BASE_Z = 0.0
ROBOT_WEST_YAW = math.radians(225.0)
ROBOT_EAST_YAW = math.radians(315.0)
ROBOT_FOOTPRINT_RADIUS = 0.37

BIN_STL_SCALE = 0.001
BIN_SIZE_X = 0.45
BIN_SIZE_Y = 0.28
BIN_SIZE_Z = 0.27
BIN_STAND_STL_SCALE = 0.001
RACK_SIZE_X = 0.70
RACK_SIZE_Y = 0.50
RACK_HEIGHT_Z = 0.72
RACK_SIDE_CLEARANCE = 0.10
# Machine 1's left-side stand is kept nearer the machine midpoint so it does
# not overlap machine 2's right-side stand in the east-bank cells.
RAW_RACK_FORWARD_OFFSET_FROM_MACHINE_CENTER = 0.05
FINISHED_RACK_FORWARD_OFFSET_FROM_MACHINE_CENTER = MACHINE_SIZE_X / 2.0 - RACK_SIZE_X / 2.0
RAW_RACK_LEFT_OFFSET_FROM_MACHINE_CENTER = MACHINE_SIZE_Y / 2.0 + RACK_SIZE_X / 2.0 + RACK_SIDE_CLEARANCE
FINISHED_RACK_AISLE_OFFSET_FROM_MACHINE_CENTER = MACHINE_SIZE_Y / 2.0 + RACK_SIZE_Y / 2.0 + RACK_SIDE_CLEARANCE
RACK_WORLD_YAW = math.pi / 2.0
# Blue and yellow bins now share the corrected yellow-bin placement and pivot.
BIN_LOCAL_CENTER_X = (-0.14, 0.14)
BIN_MESH_OFFSET_Z = -BIN_SIZE_Z / 2.0
# All bins share the corrected yellow-bin world orientation. Their vertical
# extent is the original STL Y size, so the centre sits half of it above the top.
BIN_VERTICAL_ORIGIN_Z = RACK_HEIGHT_Z + BIN_SIZE_Y / 2.0
BIN_TARGET_WORLD_YAW_DEG = 90.0

ALUMINUM_TUBE_STL_SCALE = 0.001
ALUMINUM_TUBE_DIAMETER = 0.028
ALUMINUM_TUBE_LENGTH = 0.452
HOLE_GRID_COLUMNS = 10
HOLE_GRID_ROWS = 6
HOLE_GRID_EDGE_OFFSET = 0.0225
HOLE_GRID_PITCH = 0.045
HOLE_USABLE_OUTER_RINGS = 2
TUBE_INSERTION_DEPTH = 0.275
RAW_BIN_COUNT = MACHINE_BANK_COUNT * L_PAIRS_PER_BANK * 2
USABLE_HOLES_PER_BIN = 48
TOTAL_RAW_TUBE_COUNT = RAW_BIN_COUNT * USABLE_HOLES_PER_BIN

STOCK_ZONE_INSET_X = 0.25
STOCK_ZONE_WALL_MARGIN_Y = 1.00
STOCK_ZONE_CROSS_CLEARANCE_Y = 0.35

FLOOR_THICKNESS = 0.10
WALL_THICKNESS = 0.20
WALL_HEIGHT = 3.00
WARNING_LINE_WIDTH = 0.06
AISLE_DECAL_Z = 0.0002
WARNING_DECAL_Z = 0.0004


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument(
        "--proxy-machines",
        action="store_true",
        help="Use simple boxes instead of the supplied lathe STL meshes.",
    )
    return parser.parse_args()


def room_size() -> tuple[float, float]:
    return ROOM_SIZE_X, ROOM_SIZE_Y


def asset_reference_for_output(asset_path: Path, output_path: Path) -> str:
    """Return a portable USD asset path relative to the generated scene."""
    return Path(os.path.relpath(asset_path.resolve(), output_path.parent)).as_posix()


def machine_bank_layout():
    """Return west/east banks as (name, x-parallel, x-wall, parallel-yaw, wall-yaw)."""
    room_x, _ = room_size()
    inner_wall_face_x = room_x / 2.0 - WALL_THICKNESS / 2.0
    east_wall_x = (
        inner_wall_face_x
        - WALL_MACHINE_CLEARANCE_X
        - MACHINE_SIZE_X / 2.0
    )
    east_parallel_x = east_wall_x - L_PAIR_AXIS_INTERSECTION_FROM_CENTER
    west_wall_x = -east_wall_x
    west_parallel_x = -east_parallel_x

    return [
        ("Bank_01_West", west_parallel_x, west_wall_x, math.pi / 2.0, 0.0),
        ("Bank_02_East", east_parallel_x, east_wall_x, math.pi / 2.0, math.pi),
    ]


def machine_positions() -> list[tuple[str, int, str, float, float, float]]:
    """Return 24 poses whose paired machine axes share one identical local point."""
    _, room_y = room_size()
    # Both machines use the same local workpoint (0.935, 0). With perpendicular
    # yaws this requires equal X/Y centre offsets of 1.635 m. The resulting
    # machine bodies retain the requested 0.25 m clearance on both axes.
    pair_min_y = (
        -room_y / 2.0
        + MACHINE_BANK_END_MARGIN_Y
        + L_PAIR_AXIS_INTERSECTION_FROM_CENTER
        + MACHINE_SIZE_X / 2.0
    )
    pair_max_y = room_y / 2.0 - MACHINE_BANK_END_MARGIN_Y - MACHINE_SIZE_Y / 2.0
    pair_pitch_y = (pair_max_y - pair_min_y) / (L_PAIRS_PER_BANK - 1)
    anchors_y = [pair_min_y + index * pair_pitch_y for index in range(L_PAIRS_PER_BANK)]

    poses = []
    for bank_name, parallel_x, wall_x, parallel_yaw, wall_yaw in machine_bank_layout():
        for pair_index, anchor_y in enumerate(anchors_y):
            parallel_y = anchor_y - L_PAIR_AXIS_INTERSECTION_FROM_CENTER
            pair_number = pair_index + 1
            poses.append((bank_name, pair_number, "AisleParallel", parallel_x, parallel_y, parallel_yaw))
            poses.append((bank_name, pair_number, "WallFacingAisle", wall_x, anchor_y, wall_yaw))
    return poses


def world_xy_for_local_point(pose, local_x: float, local_y: float) -> tuple[float, float]:
    """Transform a machine-local XY point using a footprint-centre pose."""
    _, _, _, center_x, center_y, yaw = pose
    relative_x = local_x - MACHINE_LOCAL_CENTER[0]
    relative_y = local_y - MACHINE_LOCAL_CENTER[1]
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (
        center_x + c * relative_x - s * relative_y,
        center_y + s * relative_x + c * relative_y,
    )


def local_xy_for_world_point(pose, world_x: float, world_y: float) -> tuple[float, float]:
    """Transform a world XY point back into the supplied machine's frame."""
    _, _, _, center_x, center_y, yaw = pose
    delta_x = world_x - center_x
    delta_y = world_y - center_y
    c = math.cos(yaw)
    s = math.sin(yaw)
    relative_x = c * delta_x + s * delta_y
    relative_y = -s * delta_x + c * delta_y
    return (
        relative_x + MACHINE_LOCAL_CENTER[0],
        relative_y + MACHINE_LOCAL_CENTER[1],
    )


def robot_workpoints() -> list[tuple[str, int, float, float, float]]:
    """Return one robot base pose at the shared axis intersection of each pair."""
    poses = machine_positions()
    workpoints = []
    for bank_name, *_ in machine_bank_layout():
        for pair_number in range(1, L_PAIRS_PER_BANK + 1):
            wall_pose = next(
                pose
                for pose in poses
                if pose[0] == bank_name and pose[1] == pair_number and pose[2] == "WallFacingAisle"
            )
            world_x, world_y = world_xy_for_local_point(
                wall_pose,
                ROBOT_WORKPOINT_LOCAL_X,
                ROBOT_WORKPOINT_LOCAL_Y,
            )
            robot_yaw = ROBOT_WEST_YAW if world_x < 0.0 else ROBOT_EAST_YAW
            workpoints.append((bank_name, pair_number, world_x, world_y, robot_yaw))
    return workpoints


def rack_positions() -> list[tuple[str, int, str, float, float, float]]:
    """Return mirrored raw racks and aisle-side finished racks for every cell."""
    poses = machine_positions()
    racks = []
    for bank_name, pair_number, *_ in robot_workpoints():
        pair = [pose for pose in poses if pose[0] == bank_name and pose[1] == pair_number]
        machine_1 = next(pose for pose in pair if pose[2] == "WallFacingAisle")
        machine_2 = next(pose for pose in pair if pose[2] == "AisleParallel")
        raw_mirrored_side_sign = -1.0 if machine_1[3] < 0.0 else 1.0
        finished_aisle_side_sign = 1.0 if machine_2[3] < 0.0 else -1.0

        for rack_role, machine, forward_offset, lateral_offset, side_sign in (
            (
                "RawMaterial",
                machine_1,
                RAW_RACK_FORWARD_OFFSET_FROM_MACHINE_CENTER,
                RAW_RACK_LEFT_OFFSET_FROM_MACHINE_CENTER,
                raw_mirrored_side_sign,
            ),
            (
                "FinishedProduct",
                machine_2,
                FINISHED_RACK_FORWARD_OFFSET_FROM_MACHINE_CENTER,
                FINISHED_RACK_AISLE_OFFSET_FROM_MACHINE_CENTER,
                finished_aisle_side_sign,
            ),
        ):
            machine_yaw = machine[5]
            front_x, front_y = math.cos(machine_yaw), math.sin(machine_yaw)
            right_x, right_y = math.sin(machine_yaw), -math.cos(machine_yaw)
            rack_x = (
                machine[3]
                + forward_offset * front_x
                + side_sign * lateral_offset * right_x
            )
            rack_y = (
                machine[4]
                + forward_offset * front_y
                + side_sign * lateral_offset * right_y
            )
            racks.append((bank_name, pair_number, rack_role, rack_x, rack_y, RACK_WORLD_YAW))
    return racks


def longitudinal_aisle_centers_x() -> tuple[float, float]:
    return LONGITUDINAL_AISLE_CENTERS_X


def usable_raw_bin_holes() -> list[tuple[int, int, float, float]]:
    """Return (row, column, local_x, local_z) for the usable outer two rings."""
    holes = []
    for row in range(HOLE_GRID_ROWS):
        for column in range(HOLE_GRID_COLUMNS):
            is_outer_ring = (
                row < HOLE_USABLE_OUTER_RINGS
                or row >= HOLE_GRID_ROWS - HOLE_USABLE_OUTER_RINGS
                or column < HOLE_USABLE_OUTER_RINGS
                or column >= HOLE_GRID_COLUMNS - HOLE_USABLE_OUTER_RINGS
            )
            if not is_outer_ring:
                continue
            local_x = (
                -BIN_SIZE_X / 2.0
                + HOLE_GRID_EDGE_OFFSET
                + column * HOLE_GRID_PITCH
            )
            local_z = (
                -BIN_SIZE_Z / 2.0
                + HOLE_GRID_EDGE_OFFSET
                + row * HOLE_GRID_PITCH
            )
            holes.append((row, column, local_x, local_z))
    return holes


def start_simulation_app(headless: bool):
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp
    return SimulationApp({"headless": headless})


def robot_asset_is_current() -> bool:
    if not ROBOT_ASSET_PATH.is_file():
        return False
    source_paths = [ROBOT_URDF_PATH, *ROBOT_MESH_DIR.glob("*.STL")]
    newest_source_mtime = max(path.stat().st_mtime for path in source_paths)
    return ROBOT_ASSET_PATH.stat().st_mtime >= newest_source_mtime


def ensure_robot_asset() -> Path:
    """Import the robot URDF once and reuse the resulting USD for all stations."""
    if robot_asset_is_current():
        return ROBOT_ASSET_PATH

    import omni.kit.app
    import omni.kit.commands
    from pxr import Usd

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
    ROBOT_ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create Isaac Sim URDF import configuration.")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.make_default_prim = True
    import_config.create_physics_scene = False
    import_config.collision_from_visuals = False

    status, _ = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(ROBOT_URDF_PATH),
        import_config=import_config,
        dest_path=str(ROBOT_ASSET_PATH),
    )
    if not status or not ROBOT_ASSET_PATH.is_file():
        raise RuntimeError(f"Failed to import robot URDF: {ROBOT_URDF_PATH}")

    asset_stage = Usd.Stage.Open(str(ROBOT_ASSET_PATH))
    if asset_stage is None or not asset_stage.GetDefaultPrim():
        raise RuntimeError(f"Imported robot asset has no default prim: {ROBOT_ASSET_PATH}")
    print(f"Imported robot asset: {ROBOT_ASSET_PATH}")
    return ROBOT_ASSET_PATH


def agv_asset_is_current() -> bool:
    if not AGV_ASSET_PATH.is_file():
        return False
    source_paths = [AGV_URDF_PATH, *AGV_MESH_DIR.glob("*.STL")]
    newest_source_mtime = max(path.stat().st_mtime for path in source_paths)
    return AGV_ASSET_PATH.stat().st_mtime >= newest_source_mtime


def ensure_agv_asset() -> Path:
    """Import the supplied HIK-Q2-400D URDF for logistics demonstrations."""
    if agv_asset_is_current():
        return AGV_ASSET_PATH

    import omni.kit.app
    import omni.kit.commands
    from pxr import Usd

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
    AGV_ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create HIK-Q2-400D import configuration")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.make_default_prim = True
    import_config.create_physics_scene = False
    import_config.collision_from_visuals = False

    status, _ = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(AGV_URDF_PATH),
        import_config=import_config,
        dest_path=str(AGV_ASSET_PATH),
    )
    if not status or not AGV_ASSET_PATH.is_file():
        raise RuntimeError(f"Failed to import AGV URDF: {AGV_URDF_PATH}")
    asset_stage = Usd.Stage.Open(str(AGV_ASSET_PATH))
    if asset_stage is None or not asset_stage.GetDefaultPrim():
        raise RuntimeError(f"Imported AGV asset has no default prim: {AGV_ASSET_PATH}")
    print(f"Imported AGV asset: {AGV_ASSET_PATH}")
    return AGV_ASSET_PATH


def create_material(stage, path: str, color):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color[:3])
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(color[3])
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(material)


def add_cube(stage, path: str, center, size, material=None, collision: bool = False):
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3d(*size))
    prim = cube.GetPrim()
    if material is not None:
        bind_material(prim, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def read_binary_stl(path: Path):
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Invalid STL file: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if expected_size != len(data):
        raise ValueError(f"Only binary STL is supported: {path}")

    points = []
    indices = []
    offset = 84
    for _ in range(triangle_count):
        values = struct.unpack_from("<12fH", data, offset)
        for vertex in (values[3:6], values[6:9], values[9:12]):
            indices.append(len(points))
            points.append(vertex)
        offset += 50
    return points, [3] * triangle_count, indices


def add_stl_mesh(
    stage,
    path: str,
    stl_path: Path,
    material=None,
    collision: bool = True,
    point_scale: float = 1.0,
    point_offset=(0.0, 0.0, 0.0),
):
    from pxr import Gf, UsdGeom, UsdPhysics

    points, face_counts, face_indices = read_binary_stl(stl_path)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(
                point[0] * point_scale + point_offset[0],
                point[1] * point_scale + point_offset[1],
                point[2] * point_scale + point_offset[2],
            )
            for point in points
        ]
    )
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    prim = mesh.GetPrim()
    if material is not None:
        bind_material(prim, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def create_aluminum_tube_asset() -> Path:
    """Create one shared USD asset from the supplied aluminum-tube STL."""
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/AluminumTube")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().CreateAttribute("factory:sourceStl", Sdf.ValueTypeNames.String).Set(
        "biped_s62/工厂场景/铝管.STL"
    )
    add_stl_mesh(
        stage,
        "/AluminumTube/Geometry",
        ALUMINUM_TUBE_STL_PATH,
        collision=True,
        point_scale=ALUMINUM_TUBE_STL_SCALE,
        point_offset=(-ALUMINUM_TUBE_DIAMETER / 2.0, 0.0, -ALUMINUM_TUBE_DIAMETER / 2.0),
    )
    ALUMINUM_TUBE_ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(ALUMINUM_TUBE_ASSET_PATH))
    return ALUMINUM_TUBE_ASSET_PATH


def root_translation_for_machine_center(x: float, y: float, yaw: float):
    local_x, local_y = MACHINE_LOCAL_CENTER
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotated_x = c * local_x - s * local_y
    rotated_y = s * local_x + c * local_y
    return x - rotated_x, y - rotated_y, -MACHINE_LOCAL_Z_MIN


def add_machine(stage, path: str, center_x: float, center_y: float, yaw: float, materials, proxy: bool):
    from pxr import Gf, Sdf, UsdGeom

    root = UsdGeom.Xform.Define(stage, path)
    root_x, root_y, root_z = root_translation_for_machine_center(center_x, center_y, yaw)
    root.AddTranslateOp().Set(Gf.Vec3d(root_x, root_y, root_z))
    root.AddRotateZOp().Set(math.degrees(yaw))
    prim = root.GetPrim()
    prim.CreateAttribute("factory:rowCenter", Sdf.ValueTypeNames.Double2).Set((center_x, center_y))
    prim.CreateAttribute("factory:sourceUrdf", Sdf.ValueTypeNames.String).Set(
        "biped_s62/工厂场景/车床/urdf/车床.urdf"
    )

    if proxy:
        # The proxy uses local coordinates so the same measured-origin correction applies.
        add_cube(
            stage,
            f"{path}/BodyProxy",
            (MACHINE_LOCAL_CENTER[0], 0.0, (MACHINE_SIZE_Z / 2.0) + MACHINE_LOCAL_Z_MIN),
            (MACHINE_SIZE_X, MACHINE_SIZE_Y, MACHINE_SIZE_Z),
            materials["machine"],
            collision=True,
        )
        return

    add_stl_mesh(
        stage,
        f"{path}/BaseLink",
        LATHE_MESH_DIR / "base_link.STL",
        materials["machine"],
    )
    add_stl_mesh(
        stage,
        f"{path}/Press",
        LATHE_MESH_DIR / "crush.STL",
        materials["machine_press"],
    )


def complement_segments(low: float, high: float, openings):
    cursor = low
    segments = []
    for opening_low, opening_high in sorted(openings):
        clipped_low = max(low, opening_low)
        clipped_high = min(high, opening_high)
        if clipped_low > cursor:
            segments.append((cursor, clipped_low))
        cursor = max(cursor, clipped_high)
    if cursor < high:
        segments.append((cursor, high))
    return segments


def add_floor_decal(stage, path: str, center, size, z: float, material) -> None:
    """Add a zero-thickness colored quad above the single collision floor."""
    from pxr import Gf, UsdGeom

    center_x, center_y = center
    size_x, size_y = size
    half_x = size_x / 2.0
    half_y = size_y / 2.0
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(center_x - half_x, center_y - half_y, z),
            Gf.Vec3f(center_x + half_x, center_y - half_y, z),
            Gf.Vec3f(center_x + half_x, center_y + half_y, z),
            Gf.Vec3f(center_x - half_x, center_y + half_y, z),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    bind_material(mesh.GetPrim(), material)


def add_floor_rectangle_border(stage, path: str, x0: float, x1: float, y0: float, y1: float, z: float, material):
    """Add a zero-thickness four-sided floor border without collision."""
    add_floor_decal(
        stage,
        f"{path}/West",
        (x0, (y0 + y1) / 2.0),
        (WARNING_LINE_WIDTH, y1 - y0),
        z,
        material,
    )
    add_floor_decal(
        stage,
        f"{path}/East",
        (x1, (y0 + y1) / 2.0),
        (WARNING_LINE_WIDTH, y1 - y0),
        z,
        material,
    )
    for side_name, y in (("South", y0), ("North", y1)):
        add_floor_decal(
            stage,
            f"{path}/{side_name}",
            ((x0 + x1) / 2.0, y),
            (x1 - x0, WARNING_LINE_WIDTH),
            z,
            material,
        )


def add_stockpile_areas(stage, materials, room_y: float, cross_x_min: float, cross_x_max: float) -> None:
    """Mark the central north/south regions as divided material stockpile bays."""
    x0 = cross_x_min + STOCK_ZONE_INSET_X
    x1 = cross_x_max - STOCK_ZONE_INSET_X
    cross_y_min = -TRANSVERSE_AISLE_WIDTH / 2.0
    cross_y_max = TRANSVERSE_AISLE_WIDTH / 2.0
    zones = (
        ("South", -room_y / 2.0 + STOCK_ZONE_WALL_MARGIN_Y, cross_y_min - STOCK_ZONE_CROSS_CLEARANCE_Y),
        ("North", cross_y_max + STOCK_ZONE_CROSS_CLEARANCE_Y, room_y / 2.0 - STOCK_ZONE_WALL_MARGIN_Y),
    )
    for zone_name, y0, y1 in zones:
        zone_path = f"/World/Factory/StockpileAreas/{zone_name}"
        add_floor_rectangle_border(
            stage,
            f"{zone_path}/Border",
            x0,
            x1,
            y0,
            y1,
            WARNING_DECAL_Z,
            materials["warning_yellow"],
        )

        # Three equal bays make future aluminium-tube and pallet placement explicit.
        for divider_index in (1, 2):
            divider_y = y0 + (y1 - y0) * divider_index / 3.0
            add_floor_decal(
                stage,
                f"{zone_path}/BayDivider_{divider_index:02d}",
                ((x0 + x1) / 2.0, divider_y),
                (x1 - x0, WARNING_LINE_WIDTH),
                WARNING_DECAL_Z,
                materials["warning_yellow"],
            )


def add_room(stage, materials) -> None:
    room_x, room_y = room_size()
    wall_z = WALL_HEIGHT / 2.0

    add_cube(
        stage,
        "/World/Factory/Floor",
        (0.0, 0.0, -FLOOR_THICKNESS / 2.0),
        (room_x, room_y, FLOOR_THICKNESS),
        materials["floor"],
        collision=True,
    )

    # Green traffic surface. The cross aisle only bridges the two longitudinal
    # lanes and no longer extends into the outer machine areas or side walls.
    west_aisle_x, east_aisle_x = longitudinal_aisle_centers_x()
    cross_x_min = west_aisle_x + LONGITUDINAL_AISLE_WIDTH / 2.0
    cross_x_max = east_aisle_x - LONGITUDINAL_AISLE_WIDTH / 2.0
    add_stockpile_areas(stage, materials, room_y, cross_x_min, cross_x_max)

    for index, aisle_x in enumerate((west_aisle_x, east_aisle_x), start=1):
        add_floor_decal(
            stage,
            f"/World/Factory/Aisles/Longitudinal_{index:02d}",
            (aisle_x, 0.0),
            (LONGITUDINAL_AISLE_WIDTH, room_y - 2.0 * WALL_THICKNESS),
            AISLE_DECAL_Z,
            materials["aisle_green"],
        )
    add_floor_decal(
        stage,
        "/World/Factory/Aisles/Transverse",
        ((cross_x_min + cross_x_max) / 2.0, 0.0),
        (cross_x_max - cross_x_min, TRANSVERSE_AISLE_WIDTH),
        AISLE_DECAL_Z,
        materials["aisle_green"],
    )

    # Yellow warning lines mark every aisle/work-area boundary.  Inner lines
    # are interrupted at the cross aisle so the three lanes read as connected.
    longitudinal_length = room_y - 2.0 * WALL_THICKNESS
    longitudinal_y_min = -longitudinal_length / 2.0
    longitudinal_y_max = longitudinal_length / 2.0
    cross_y_min = -TRANSVERSE_AISLE_WIDTH / 2.0
    cross_y_max = TRANSVERSE_AISLE_WIDTH / 2.0

    outer_edges_x = (
        west_aisle_x - LONGITUDINAL_AISLE_WIDTH / 2.0,
        east_aisle_x + LONGITUDINAL_AISLE_WIDTH / 2.0,
    )
    for index, edge_x in enumerate(outer_edges_x, start=1):
        add_floor_decal(
            stage,
            f"/World/Factory/WarningLines/LongitudinalOuter_{index:02d}",
            (edge_x, 0.0),
            (WARNING_LINE_WIDTH, longitudinal_length),
            WARNING_DECAL_Z,
            materials["warning_yellow"],
        )

    inner_edges_x = (cross_x_min, cross_x_max)
    y_segments = (
        (longitudinal_y_min, cross_y_min),
        (cross_y_max, longitudinal_y_max),
    )
    for edge_index, edge_x in enumerate(inner_edges_x, start=1):
        for segment_index, (y0, y1) in enumerate(y_segments, start=1):
            add_floor_decal(
                stage,
                f"/World/Factory/WarningLines/LongitudinalInner_{edge_index:02d}_{segment_index:02d}",
                (edge_x, (y0 + y1) / 2.0),
                (WARNING_LINE_WIDTH, y1 - y0),
                WARNING_DECAL_Z,
                materials["warning_yellow"],
            )

    for index, edge_y in enumerate((cross_y_min, cross_y_max), start=1):
        add_floor_decal(
            stage,
            f"/World/Factory/WarningLines/Transverse_{index:02d}",
            ((cross_x_min + cross_x_max) / 2.0, edge_y),
            (cross_x_max - cross_x_min, WARNING_LINE_WIDTH),
            WARNING_DECAL_Z,
            materials["warning_yellow"],
        )

    # Only the two longitudinal lanes pass through the north and south walls.
    x_openings = [
        (x - LONGITUDINAL_AISLE_WIDTH / 2.0, x + LONGITUDINAL_AISLE_WIDTH / 2.0)
        for x in longitudinal_aisle_centers_x()
    ]
    for side_name, wall_y in (("South", -room_y / 2.0), ("North", room_y / 2.0)):
        for index, (x0, x1) in enumerate(complement_segments(-room_x / 2.0, room_x / 2.0, x_openings), start=1):
            add_cube(
                stage,
                f"/World/Factory/Walls/{side_name}_{index:02d}",
                ((x0 + x1) / 2.0, wall_y, wall_z),
                (x1 - x0, WALL_THICKNESS, WALL_HEIGHT),
                materials["wall"],
                collision=True,
            )

    # The transverse lane ends at the longitudinal lanes, so both side walls
    # are continuous and no longer contain a central opening.
    for side_name, wall_x in (("West", -room_x / 2.0), ("East", room_x / 2.0)):
        add_cube(
            stage,
            f"/World/Factory/Walls/{side_name}",
            (wall_x, 0.0, wall_z),
            (WALL_THICKNESS, room_y, WALL_HEIGHT),
            materials["wall"],
            collision=True,
        )


def add_machines(stage, materials, proxy: bool) -> None:
    from pxr import Sdf

    for bank_name, pair_number, machine_role, x, y, yaw in machine_positions():
        machine_number = 1 if machine_role == "WallFacingAisle" else 2
        machine_name = "Machine_01_Wall_Process01" if machine_number == 1 else "Machine_02_Aisle_Process02"
        path = f"/World/Factory/Machines/{bank_name}/Cell_{pair_number:02d}/{machine_name}"
        add_machine(
            stage,
            path,
            x,
            y,
            yaw,
            materials,
            proxy,
        )
        prim = stage.GetPrimAtPath(path)
        prim.CreateAttribute("factory:cellNumber", Sdf.ValueTypeNames.Int).Set(pair_number)
        prim.CreateAttribute("factory:machineNumber", Sdf.ValueTypeNames.Int).Set(machine_number)
        prim.CreateAttribute("factory:processStep", Sdf.ValueTypeNames.Int).Set(machine_number)


def add_robot_instances(stage, robot_asset_path: Path | str) -> None:
    """Reference one robot USD at each paired-machine workpoint."""
    from pxr import Gf, Sdf, UsdGeom

    UsdGeom.Xform.Define(stage, "/World/Factory/Robots")
    for robot_index, (bank_name, pair_number, x, y, yaw) in enumerate(robot_workpoints(), start=1):
        side_name = "West" if x < 0.0 else "East"
        path = f"/World/Factory/Robots/Robot_{robot_index:02d}_{side_name}_Pair_{pair_number:02d}"
        station = UsdGeom.Xform.Define(stage, path)
        station.AddTranslateOp().Set(Gf.Vec3d(x, y, ROBOT_BASE_Z))
        station.AddRotateZOp().Set(math.degrees(yaw))
        station_prim = station.GetPrim()
        station_prim.CreateAttribute("factory:machineBank", Sdf.ValueTypeNames.String).Set(bank_name)
        station_prim.CreateAttribute("factory:machinePair", Sdf.ValueTypeNames.Int).Set(pair_number)
        station_prim.CreateAttribute("factory:workpointLocalXY", Sdf.ValueTypeNames.Double2).Set(
            (ROBOT_WORKPOINT_LOCAL_X, ROBOT_WORKPOINT_LOCAL_Y)
        )
        # Keep placement ops on the station parent so they never conflict with
        # xform ops authored on the imported robot asset's default prim.
        model_prim = stage.OverridePrim(f"{path}/Model")
        model_prim.GetReferences().AddReference(str(robot_asset_path))


def add_raw_bin_tubes(stage, orientation_path: str, tube_asset_path: Path | str, material) -> None:
    """Insert one shared-asset tube into each usable hole of a raw-material bin."""
    from pxr import Gf, Sdf, UsdGeom

    tubes_path = f"{orientation_path}/Tubes"
    UsdGeom.Xform.Define(stage, tubes_path)
    tube_start_y = BIN_SIZE_Y / 2.0 - TUBE_INSERTION_DEPTH
    for row, column, local_x, local_z in usable_raw_bin_holes():
        tube_path = f"{tubes_path}/Tube_R{row + 1:02d}_C{column + 1:02d}"
        tube = UsdGeom.Xform.Define(stage, tube_path)
        tube.AddTranslateOp().Set(Gf.Vec3d(local_x, tube_start_y, local_z))
        tube_prim = tube.GetPrim()
        tube_prim.CreateAttribute("factory:holeRow", Sdf.ValueTypeNames.Int).Set(row + 1)
        tube_prim.CreateAttribute("factory:holeColumn", Sdf.ValueTypeNames.Int).Set(column + 1)
        tube_prim.CreateAttribute("factory:insertionDepthMeters", Sdf.ValueTypeNames.Double).Set(
            TUBE_INSERTION_DEPTH
        )
        model_prim = stage.OverridePrim(f"{tube_path}/Model")
        model_prim.GetReferences().AddReference(str(tube_asset_path))
        bind_material(model_prim, material)
        model_prim.SetInstanceable(True)


def add_bin_rack(
    stage,
    path: str,
    x: float,
    y: float,
    yaw: float,
    rack_role: str,
    materials,
    tube_asset_path: Path,
) -> None:
    """Load the supplied two-bay bin stand and place one supplied bin in each bay."""
    from pxr import Gf, Sdf, UsdGeom

    rack = UsdGeom.Xform.Define(stage, path)
    rack.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
    rack.AddRotateZOp().Set(math.degrees(yaw))
    rack_prim = rack.GetPrim()
    rack_prim.CreateAttribute("factory:rackRole", Sdf.ValueTypeNames.String).Set(rack_role)
    rack_prim.CreateAttribute("factory:binCount", Sdf.ValueTypeNames.Int).Set(2)

    rack_prim.CreateAttribute("factory:sourceStl", Sdf.ValueTypeNames.String).Set(
        "biped_s62/工厂场景/料箱座.STL"
    )
    stand = UsdGeom.Xform.Define(stage, f"{path}/StandMesh")
    # Source STL axes are X=width, Y=height, Z=depth. Rotate about X so Y is up.
    stand.AddTranslateOp().Set(Gf.Vec3d(0.0, RACK_SIZE_Y / 2.0, 0.0))
    stand.AddRotateXOp().Set(90.0)
    add_stl_mesh(
        stage,
        f"{path}/StandMesh/Geometry",
        BIN_STAND_STL_PATH,
        materials["rack_frame"],
        collision=True,
        point_scale=BIN_STAND_STL_SCALE,
        point_offset=(-RACK_SIZE_X / 2.0, 0.0, 0.0),
    )

    bin_material = materials["raw_bin"] if rack_role == "RawMaterial" else materials["finished_bin"]
    for bin_index, bin_center_x in enumerate(BIN_LOCAL_CENTER_X, start=1):
        bin_root = UsdGeom.Xform.Define(stage, f"{path}/Bins/Bin_{bin_index:02d}")
        bin_root.AddTranslateOp().Set(
            Gf.Vec3d(bin_center_x, 0.0, BIN_VERTICAL_ORIGIN_Z)
        )

        # Rack yaws differ by bank and process. Cancel that difference so every
        # blue and yellow bin has exactly the corrected yellow-bin world yaw.
        yaw_alignment = UsdGeom.Xform.Define(
            stage, f"{path}/Bins/Bin_{bin_index:02d}/WorldYawAlignment"
        )
        yaw_alignment.AddRotateZOp().Set(
            BIN_TARGET_WORLD_YAW_DEG - math.degrees(yaw)
        )

        # In the aligned frame, local Y is the world X axis perpendicular to
        # the aisle. The shared -90-degree tilt makes every hole face vertical
        # and point toward the same direction along the aisle.
        aisle_axis_tilt = UsdGeom.Xform.Define(
            stage, f"{path}/Bins/Bin_{bin_index:02d}/WorldYawAlignment/AisleAxisTilt"
        )
        aisle_axis_tilt.AddRotateYOp().Set(-90.0)

        orientation_path = (
            f"{path}/Bins/Bin_{bin_index:02d}/WorldYawAlignment/AisleAxisTilt/Orientation"
        )
        orientation = UsdGeom.Xform.Define(stage, orientation_path)
        orientation.AddRotateZOp().Set(90.0)
        orientation.AddRotateXOp().Set(180.0)
        add_stl_mesh(
            stage,
            f"{orientation_path}/Geometry",
            BIN_STL_PATH,
            bin_material,
            collision=True,
            point_scale=BIN_STL_SCALE,
            point_offset=(-BIN_SIZE_X / 2.0, -BIN_SIZE_Y / 2.0, BIN_MESH_OFFSET_Z),
        )
        if rack_role == "RawMaterial":
            add_raw_bin_tubes(
                stage,
                orientation_path,
                tube_asset_path,
                materials["aluminum_tube"],
            )


def add_material_racks(stage, materials, tube_asset_path: Path | str) -> None:
    """Add one raw rack and one finished rack to every paired-machine cell."""
    from pxr import Sdf, UsdGeom

    UsdGeom.Xform.Define(stage, "/World/Factory/MaterialRacks")
    for bank_name, pair_number, rack_role, x, y, yaw in rack_positions():
        cell_path = f"/World/Factory/MaterialRacks/{bank_name}/Cell_{pair_number:02d}"
        rack_name = "RawMaterialRack_Machine01_MirroredSide" if rack_role == "RawMaterial" else "FinishedRack_Machine02_AisleSide"
        rack_path = f"{cell_path}/{rack_name}"
        add_bin_rack(stage, rack_path, x, y, yaw, rack_role, materials, tube_asset_path)
        rack_prim = stage.GetPrimAtPath(rack_path)
        rack_prim.CreateAttribute("factory:cellNumber", Sdf.ValueTypeNames.Int).Set(pair_number)
        rack_prim.CreateAttribute("factory:servesMachine", Sdf.ValueTypeNames.Int).Set(
            1 if rack_role == "RawMaterial" else 2
        )


def add_lighting_and_camera(stage) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(450.0)

    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(1800.0)
    key.CreateAngleAttr(0.6)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, -25.0, 25.0))

    fill = UsdLux.RectLight.Define(stage, "/World/Lights/CeilingFill")
    fill.CreateIntensityAttr(900.0)
    fill.CreateWidthAttr(14.0)
    fill.CreateHeightAttr(20.0)
    fill.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 8.0))
    fill.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    eye = Gf.Vec3d(22.0, -30.0, 24.0)
    target = Gf.Vec3d(0.0, 0.0, 0.8)
    camera.AddTransformOp().Set(Gf.Matrix4d(1.0).SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse())
    camera.CreateFocalLengthAttr(22.0)


def validate_layout() -> None:
    room_x, room_y = room_size()
    poses = machine_positions()
    usable_holes = usable_raw_bin_holes()
    assert MACHINE_BANK_COUNT == 2
    assert MACHINES_PER_BANK == 12
    assert len(poses) == TOTAL_MACHINE_COUNT == 24
    assert len(machine_bank_layout()) == MACHINE_BANK_COUNT
    assert len(robot_workpoints()) == MACHINE_BANK_COUNT * L_PAIRS_PER_BANK == 12
    assert len(rack_positions()) == 24
    assert sum(rack[2] == "RawMaterial" for rack in rack_positions()) == 12
    assert sum(rack[2] == "FinishedProduct" for rack in rack_positions()) == 12
    assert HOLE_GRID_COLUMNS * HOLE_GRID_ROWS == 60
    assert len(usable_holes) == USABLE_HOLES_PER_BIN == 48
    assert HOLE_GRID_COLUMNS * HOLE_GRID_ROWS - len(usable_holes) == 12
    assert RAW_BIN_COUNT == 24
    assert TOTAL_RAW_TUBE_COUNT == 1152
    assert abs(usable_holes[0][2] + BIN_SIZE_X / 2.0 - HOLE_GRID_EDGE_OFFSET) < 1e-9
    assert abs(usable_holes[0][3] + BIN_SIZE_Z / 2.0 - HOLE_GRID_EDGE_OFFSET) < 1e-9
    assert abs(ALUMINUM_TUBE_LENGTH - 0.452) < 1e-9
    assert 0.0 < TUBE_INSERTION_DEPTH < ALUMINUM_TUBE_LENGTH
    assert 0.0 <= BIN_SIZE_Y - TUBE_INSERTION_DEPTH
    assert abs(sum(BIN_LOCAL_CENTER_X)) < 1e-9
    assert abs(BIN_MESH_OFFSET_Z + BIN_SIZE_Z / 2.0) < 1e-9
    assert abs(
        BIN_LOCAL_CENTER_X[1]
        - BIN_LOCAL_CENTER_X[0]
        - BIN_SIZE_Z
        - 0.01
    ) < 1e-9
    assert all(abs(rack[5] - RACK_WORLD_YAW) < 1e-9 for rack in rack_positions())
    assert abs(BIN_VERTICAL_ORIGIN_Z - BIN_SIZE_Y / 2.0 - RACK_HEIGHT_Z) < 1e-9
    assert abs(LONGITUDINAL_AISLE_WIDTH - 2.5) < 1e-9
    assert abs(TRANSVERSE_AISLE_WIDTH - 2.5) < 1e-9
    assert all(abs(x) > abs(LONGITUDINAL_AISLE_CENTERS_X[0]) for _, _, _, x, _, _ in poses)
    assert all(abs(math.cos(yaw)) < 1e-9 for _, _, role, _, _, yaw in poses if role == "AisleParallel")
    assert all(abs(math.sin(yaw)) < 1e-9 for _, _, role, _, _, yaw in poses if role == "WallFacingAisle")
    inner_wall_face_x = room_x / 2.0 - WALL_THICKNESS / 2.0
    wall_poses = [pose for pose in poses if pose[2] == "WallFacingAisle"]
    assert all(
        abs(inner_wall_face_x - (abs(pose[3]) + MACHINE_SIZE_X / 2.0) - WALL_MACHINE_CLEARANCE_X) < 1e-9
        for pose in wall_poses
    )
    for bank_name, *_ in machine_bank_layout():
        for pair_number in range(1, L_PAIRS_PER_BANK + 1):
            pair = [pose for pose in poses if pose[0] == bank_name and pose[1] == pair_number]
            parallel = next(pose for pose in pair if pose[2] == "AisleParallel")
            wall = next(pose for pose in pair if pose[2] == "WallFacingAisle")
            wall_workpoint = world_xy_for_local_point(
                wall, ROBOT_WORKPOINT_LOCAL_X, ROBOT_WORKPOINT_LOCAL_Y
            )
            parallel_workpoint = world_xy_for_local_point(
                parallel, ROBOT_WORKPOINT_LOCAL_X, ROBOT_WORKPOINT_LOCAL_Y
            )
            assert math.dist(wall_workpoint, parallel_workpoint) < 1e-9
            for pose in (parallel, wall):
                recovered_local = local_xy_for_world_point(pose, *wall_workpoint)
                assert math.dist(
                    recovered_local,
                    (ROBOT_WORKPOINT_LOCAL_X, ROBOT_WORKPOINT_LOCAL_Y),
                ) < 1e-9

            parallel_front_y = parallel[4] + MACHINE_SIZE_X / 2.0
            wall_body_min_y = wall[4] - MACHINE_SIZE_Y / 2.0
            assert abs(wall_body_min_y - parallel_front_y - L_PAIR_MACHINE_GAP_X) < 1e-9
            if wall[3] < 0.0:
                wall_front_x = wall[3] + MACHINE_SIZE_X / 2.0
                parallel_wallward_x = parallel[3] - MACHINE_SIZE_Y / 2.0
                actual_gap_x = parallel_wallward_x - wall_front_x
            else:
                wall_front_x = wall[3] - MACHINE_SIZE_X / 2.0
                parallel_wallward_x = parallel[3] + MACHINE_SIZE_Y / 2.0
                actual_gap_x = wall_front_x - parallel_wallward_x
            assert abs(actual_gap_x - L_PAIR_MACHINE_GAP_X) < 1e-9

    def half_extents(size_x: float, size_y: float, yaw: float):
        return (
            abs(math.cos(yaw)) * size_x / 2.0 + abs(math.sin(yaw)) * size_y / 2.0,
            abs(math.sin(yaw)) * size_x / 2.0 + abs(math.cos(yaw)) * size_y / 2.0,
        )

    racks = rack_positions()
    for rack in racks:
        serving_role = "WallFacingAisle" if rack[2] == "RawMaterial" else "AisleParallel"
        serving_machine = next(
            pose
            for pose in poses
            if pose[0] == rack[0] and pose[1] == rack[1] and pose[2] == serving_role
        )
        rack_local_x, rack_local_y = local_xy_for_world_point(
            serving_machine, rack[3], rack[4]
        )
        expected_forward = (
            RAW_RACK_FORWARD_OFFSET_FROM_MACHINE_CENTER
            if rack[2] == "RawMaterial"
            else FINISHED_RACK_FORWARD_OFFSET_FROM_MACHINE_CENTER
        )
        expected_local_y = (
            (
                MACHINE_LOCAL_CENTER[1] + RAW_RACK_LEFT_OFFSET_FROM_MACHINE_CENTER
                if serving_machine[3] < 0.0
                else MACHINE_LOCAL_CENTER[1] - RAW_RACK_LEFT_OFFSET_FROM_MACHINE_CENTER
            )
            if rack[2] == "RawMaterial"
            else (
                MACHINE_LOCAL_CENTER[1] - FINISHED_RACK_AISLE_OFFSET_FROM_MACHINE_CENTER
                if serving_machine[3] < 0.0
                else MACHINE_LOCAL_CENTER[1] + FINISHED_RACK_AISLE_OFFSET_FROM_MACHINE_CENTER
            )
        )
        assert abs(rack_local_x - MACHINE_LOCAL_CENTER[0] - expected_forward) < 1e-9
        assert abs(rack_local_y - expected_local_y) < 1e-9

        rack_hx, rack_hy = half_extents(RACK_SIZE_X, RACK_SIZE_Y, rack[5])
        for machine in poses:
            machine_hx, machine_hy = half_extents(MACHINE_SIZE_X, MACHINE_SIZE_Y, machine[5])
            overlaps = (
                abs(rack[3] - machine[3]) < rack_hx + machine_hx
                and abs(rack[4] - machine[4]) < rack_hy + machine_hy
            )
            assert not overlaps
        robot = next(
            item for item in robot_workpoints() if item[0] == rack[0] and item[1] == rack[1]
        )
        rack_radius = math.hypot(RACK_SIZE_X / 2.0, RACK_SIZE_Y / 2.0)
        assert math.hypot(rack[3] - robot[2], rack[4] - robot[3]) > ROBOT_FOOTPRINT_RADIUS + rack_radius

    for pair_number in range(1, L_PAIRS_PER_BANK + 1):
        west_raw = next(
            rack
            for rack in racks
            if rack[0] == "Bank_01_West"
            and rack[1] == pair_number
            and rack[2] == "RawMaterial"
        )
        east_raw = next(
            rack
            for rack in racks
            if rack[0] == "Bank_02_East"
            and rack[1] == pair_number
            and rack[2] == "RawMaterial"
        )
        west_finished = next(
            rack
            for rack in racks
            if rack[0] == "Bank_01_West"
            and rack[1] == pair_number
            and rack[2] == "FinishedProduct"
        )
        east_finished = next(
            rack
            for rack in racks
            if rack[0] == "Bank_02_East"
            and rack[1] == pair_number
            and rack[2] == "FinishedProduct"
        )
        assert abs(west_raw[3] + east_raw[3]) < 1e-9
        assert abs(west_raw[4] - east_raw[4]) < 1e-9
        assert abs(west_finished[3] + east_finished[3]) < 1e-9
        assert abs(west_finished[4] - east_finished[4]) < 1e-9

    for rack_index, rack in enumerate(racks):
        rack_hx, rack_hy = half_extents(RACK_SIZE_X, RACK_SIZE_Y, rack[5])
        for other in racks[rack_index + 1 :]:
            other_hx, other_hy = half_extents(RACK_SIZE_X, RACK_SIZE_Y, other[5])
            overlaps = (
                abs(rack[3] - other[3]) < rack_hx + other_hx
                and abs(rack[4] - other[4]) < rack_hy + other_hy
            )
            assert not overlaps
    assert room_x > 0.0 and room_y > 0.0


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    validate_layout()
    app = start_simulation_app(args.headless)

    try:
        from pxr import Sdf, Usd, UsdGeom

        robot_asset_path = ensure_robot_asset()
        aluminum_tube_asset_path = create_aluminum_tube_asset()
        robot_asset_reference = asset_reference_for_output(robot_asset_path, args.output)
        aluminum_tube_reference = asset_reference_for_output(
            aluminum_tube_asset_path, args.output
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stage = Usd.Stage.CreateNew(str(args.output))
        if stage is None:
            raise RuntimeError(f"Could not create output stage: {args.output}")
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(240)
        stage.SetTimeCodesPerSecond(24)

        world = stage.GetPrimAtPath("/World")
        room_x, room_y = room_size()
        world.CreateAttribute("factory:roomSizeMeters", Sdf.ValueTypeNames.Double2).Set((room_x, room_y))
        world.CreateAttribute("factory:machineCount", Sdf.ValueTypeNames.Int).Set(TOTAL_MACHINE_COUNT)
        world.CreateAttribute("factory:machineBankCount", Sdf.ValueTypeNames.Int).Set(MACHINE_BANK_COUNT)
        world.CreateAttribute("factory:machinesPerBank", Sdf.ValueTypeNames.Int).Set(MACHINES_PER_BANK)
        world.CreateAttribute("factory:robotWorkpointCount", Sdf.ValueTypeNames.Int).Set(
            MACHINE_BANK_COUNT * L_PAIRS_PER_BANK
        )
        world.CreateAttribute("factory:robotWorkpointLocalXY", Sdf.ValueTypeNames.Double2).Set(
            (ROBOT_WORKPOINT_LOCAL_X, ROBOT_WORKPOINT_LOCAL_Y)
        )
        world.CreateAttribute("factory:robotAsset", Sdf.ValueTypeNames.String).Set(
            robot_asset_reference
        )
        world.CreateAttribute("factory:materialRackCount", Sdf.ValueTypeNames.Int).Set(24)
        world.CreateAttribute("factory:materialBinCount", Sdf.ValueTypeNames.Int).Set(48)
        world.CreateAttribute("factory:rawTubeCount", Sdf.ValueTypeNames.Int).Set(
            TOTAL_RAW_TUBE_COUNT
        )
        world.CreateAttribute("factory:usableHolesPerRawBin", Sdf.ValueTypeNames.Int).Set(
            USABLE_HOLES_PER_BIN
        )
        world.CreateAttribute("factory:aluminumTubeAsset", Sdf.ValueTypeNames.String).Set(
            aluminum_tube_reference
        )
        world.CreateAttribute("factory:wallMachineClearanceMeters", Sdf.ValueTypeNames.Double).Set(
            WALL_MACHINE_CLEARANCE_X
        )
        world.CreateAttribute("factory:longitudinalAisleWidthMeters", Sdf.ValueTypeNames.Double).Set(
            LONGITUDINAL_AISLE_WIDTH
        )
        world.CreateAttribute("factory:transverseAisleWidthMeters", Sdf.ValueTypeNames.Double).Set(
            TRANSVERSE_AISLE_WIDTH
        )

        materials = {
            "floor": create_material(stage, "/World/Materials/WorkAreaGray", (0.38, 0.39, 0.40, 1.0)),
            "wall": create_material(stage, "/World/Materials/Wall", (0.82, 0.83, 0.80, 1.0)),
            "aisle_green": create_material(stage, "/World/Materials/AisleGreen", (0.03, 0.48, 0.14, 1.0)),
            "warning_yellow": create_material(stage, "/World/Materials/WarningYellow", (0.96, 0.72, 0.03, 1.0)),
            # Industrial safety-yellow enclosure.  The press/mould keeps its
            # separate dark material (and may still be recolored by previews).
            "machine": create_material(stage, "/World/Materials/MachineBody", (0.96, 0.62, 0.02, 1.0)),
            "machine_press": create_material(stage, "/World/Materials/MachinePress", (0.23, 0.29, 0.31, 1.0)),
            "rack_frame": create_material(stage, "/World/Materials/RackFrame", (0.16, 0.18, 0.20, 1.0)),
            "raw_bin": create_material(stage, "/World/Materials/RawMaterialBin", (0.08, 0.30, 0.82, 1.0)),
            "finished_bin": create_material(stage, "/World/Materials/FinishedProductBin", (0.92, 0.38, 0.06, 1.0)),
            "aluminum_tube": create_material(stage, "/World/Materials/AluminumTube", (0.70, 0.73, 0.75, 1.0)),
        }

        add_room(stage, materials)
        add_machines(stage, materials, args.proxy_machines)
        add_material_racks(stage, materials, aluminum_tube_reference)
        add_robot_instances(stage, robot_asset_reference)
        add_lighting_and_camera(stage)

        stage.GetRootLayer().Save()

        print(f"Saved scene: {args.output}")
        print(f"Room size: {room_x:.2f} m x {room_y:.2f} m")
        print("Machine layout: 2 continuous banks x 12 = 24 lathes; six evenly spaced L pairs per side")
        print("L-pair orientation: aisle-side machines are parallel; wall-side machines face aisles")
        print(
            "Robot work corners: paired centreline intersection is local "
            f"({ROBOT_WORKPOINT_LOCAL_X:.3f}, {ROBOT_WORKPOINT_LOCAL_Y:.3f}) m in both machine frames"
        )
        pair_anchors = [
            pose[4]
            for pose in machine_positions()
            if pose[0] == "Bank_01_West" and pose[2] == "WallFacingAisle"
        ]
        pair_pitch = pair_anchors[1] - pair_anchors[0]
        aisle_outer_edge_x = abs(LONGITUDINAL_AISLE_CENTERS_X[1]) + LONGITUDINAL_AISLE_WIDTH / 2.0
        parallel_x = abs(next(pose[3] for pose in machine_positions() if pose[2] == "AisleParallel"))
        aisle_clearance = parallel_x - MACHINE_SIZE_Y / 2.0 - aisle_outer_edge_x
        print(
            f"Clearances: wall {WALL_MACHINE_CLEARANCE_X:.3f} m; aisle {aisle_clearance:.3f} m; "
            f"L-pair pitch {pair_pitch:.3f} m"
        )
        print(f"Robots: {len(robot_workpoints())} independent references placed at paired workpoints")
        print("Material handling: 24 supplied STL stands, 48 supplied STL bins (24 raw + 24 finished)")
        print(
            f"Raw tubes: {TOTAL_RAW_TUBE_COUNT} shared-asset instances; "
            f"{USABLE_HOLES_PER_BIN} usable outer-ring holes per raw bin"
        )
        print("Central layout: north and south stockpile areas, three marked bays each; no central machines")
        print(
            f"Green aisles: two longitudinal aisles, {LONGITUDINAL_AISLE_WIDTH:.2f} m wide; "
            f"one transverse aisle, {TRANSVERSE_AISLE_WIDTH:.2f} m wide"
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
