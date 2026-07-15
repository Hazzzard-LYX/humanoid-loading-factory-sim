#!/usr/bin/env python3
"""Demonstrate two HIK-Q2-400D AGVs scheduling each three-cell pipeline."""

from __future__ import annotations

import argparse
import math
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from create_factory_scene import (
    AGV_MESH_DIR,
    LONGITUDINAL_AISLE_CENTERS_X,
    RACK_WORLD_YAW,
    ROOM_SIZE_Y,
    add_stl_mesh,
    create_material,
    rack_positions,
)
from preview_all_cells_loading import (
    build_cell_specs,
    mirror_active,
    solve_all_cycles,
    transform_base,
    transform_tube_pose,
)
from preview_single_robot_loading import (
    BASE_GROUND_Z,
    CONTINUOUS_PICKUP_SCHEDULE,
    SCENE_PATH,
    TUBE_ASSET_PATH,
    finger_and_tube_transform,
    full_joint_vector,
    interpolate_base,
    smoothstep,
)


DEMO_ROOT = "/World/AGVLogisticsDemo"
AGV_SPEED = 1.35
AGV_ANGULAR_SPEED = math.radians(75.0)
PICKUP_DWELL = 1.0
DROP_DWELL = 1.0
AGV_LIFT_TRAVEL = 0.06
AISLE_LANE_OFFSET = 0.55
ROBOT_CLEARANCE_DELAY = 2.0
ROBOT_TIME_SCALE = 1.0
FINISHED_TUBES_PER_RACK = 96
AGV2_INITIAL_STAGGER = 4.0
STATION_MOUTH_CLEARANCE = 0.75
RAW_STOCK_APPROACH_CLEARANCE = 1.15
AGV_HALF_EXTENTS = np.array([0.39, 0.2725])
RACK_HALF_EXTENTS = np.array([0.25, 0.35])
TRAFFIC_SAFETY_MARGIN = 0.04
# Cooperative V2V reservations may reroute one task to the other lane.
TASK_LANE_OVERRIDES: dict[tuple[int, int], float] = {
    # North-line AGV-2 changes to the other lane for the full-stock delivery,
    # then returns to its normal lane while clearing the destination station.
    (2, 6): -1.0,
    (4, 6): -1.0,
}
COOPERATIVE_ENTRY_HOLDS: dict[int, dict[int, float]] = {
    # V2V waits happen only after the rack has left the workstation mouth.
    # All four lines still dispatch AGV-1/AGV-2 at t=0/t=4 seconds.
    2: {0: 130.0, 1: 130.0},
    3: {4: 8.0, 6: 8.0},
    4: {0: 145.0, 1: 145.0, 4: 8.0, 6: 8.0},
}
COOPERATIVE_DEADHEAD_HOLDS: dict[int, dict[int, float]] = {
    # These vehicles receive the V2V message before entering an occupied
    # intersection, so they remain stopped without lateral sliding.
    3: {1: 0.5},
    4: {1: 0.5, 6: 15.0},
}


@dataclass(frozen=True)
class PipelineSpec:
    index: int
    bank: str
    pairs: tuple[int, int, int]
    aisle_x: float
    exit_y: float
    raw_stock_xy: tuple[float, float]
    finished_stock_xy: tuple[float, float]
    parking_xy: tuple[tuple[float, float], tuple[float, float]]
    start_delay: float = 0.0


@dataclass(frozen=True)
class Task:
    index: int
    name: str
    pickup: tuple[float, float]
    dropoff: tuple[float, float]
    rack_path: str
    dependencies: tuple[int, ...] = ()
    vacancy_dependency: int | None = None
    trigger_pair: int | None = None
    disappear: bool = False
    pickup_rack_yaw: float = RACK_WORLD_YAW
    dropoff_rack_yaw: float = RACK_WORLD_YAW


@dataclass
class ScheduledTask:
    task: Task
    agv_index: int
    start_position: tuple[float, float]
    start_time: float
    pickup_arrival: float
    pickup_departure: float
    drop_arrival: float
    drop_complete: float
    station_clear_time: float
    end_time: float
    deadhead_route: list[tuple[float, float]]
    loaded_route: list[tuple[float, float]]
    clearance_route: list[tuple[float, float]]
    travel_distance: float
    loaded_hold_duration: float = 0.0
    deadhead_hold_duration: float = 0.0


@dataclass
class Plan:
    pipeline: PipelineSpec
    tasks: list[ScheduledTask]
    agv_start_positions: tuple[tuple[float, float], tuple[float, float]]
    robot_trigger_times: dict[int, float]
    makespan: float
    total_distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def pipeline_specs() -> list[PipelineSpec]:
    return [
        PipelineSpec(1, "Bank_01_West", (1, 2, 3), LONGITUDINAL_AISLE_CENTERS_X[0], -ROOM_SIZE_Y / 2.0 + 0.35, (-1.20, 3.5), (-1.20, -8.5), ((-0.55, -7.0), (0.55, -7.0)), 0.0),
        PipelineSpec(2, "Bank_01_West", (4, 5, 6), LONGITUDINAL_AISLE_CENTERS_X[0], ROOM_SIZE_Y / 2.0 - 0.35, (-1.20, 8.5), (-1.20, -10.5), ((-0.55, -3.4), (0.55, -3.4)), 0.0),
        PipelineSpec(3, "Bank_02_East", (1, 2, 3), LONGITUDINAL_AISLE_CENTERS_X[1], -ROOM_SIZE_Y / 2.0 + 0.35, (1.20, 3.5), (1.20, -8.5), ((-0.55, -5.2), (0.55, -5.2)), 0.0),
        PipelineSpec(4, "Bank_02_East", (4, 5, 6), LONGITUDINAL_AISLE_CENTERS_X[1], ROOM_SIZE_Y / 2.0 - 0.35, (1.20, 8.5), (1.20, -10.5), ((-0.55, -1.6), (0.55, -1.6)), 0.0),
    ]


def rack_path(bank: str, pair: int, raw: bool) -> str:
    name = (
        "RawMaterialRack_Machine01_MirroredSide"
        if raw
        else "FinishedRack_Machine02_AisleSide"
    )
    return f"/World/Factory/MaterialRacks/{bank}/Cell_{pair:02d}/{name}"


def rack_pose_map():
    return {
        (bank, pair, role): (x, y)
        for bank, pair, role, x, y, _yaw in rack_positions()
    }


def make_route(
    start: tuple[float, float],
    end: tuple[float, float],
    lane_x: float,
) -> list[tuple[float, float]]:
    points = [start]
    candidates = [(lane_x, start[1]), (lane_x, end[1]), end]
    for point in candidates:
        if math.dist(points[-1], point) > 1e-6:
            points.append(point)
    return points


def make_raw_stock_pickup_routes(
    start: tuple[float, float],
    pickup: tuple[float, float],
    dropoff: tuple[float, float],
    lane_x: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Approach a stock rack through its open long side, never through a row."""
    outside_y = pickup[1] - RAW_STOCK_APPROACH_CLEARANCE
    deadhead = [start]
    for point in (
        (lane_x, start[1]),
        (lane_x, outside_y),
        (pickup[0], outside_y),
        pickup,
    ):
        if math.dist(deadhead[-1], point) > 1e-6:
            deadhead.append(point)
    loaded = [pickup]
    for point in (
        (pickup[0], outside_y),
        (lane_x, outside_y),
        (lane_x, dropoff[1]),
        dropoff,
    ):
        if math.dist(loaded[-1], point) > 1e-6:
            loaded.append(point)
    return deadhead, loaded


def traffic_lane_x(
    pipeline: PipelineSpec,
    start: tuple[float, float],
    end: tuple[float, float],
    agv_index: int,
    task_index: int | None = None,
) -> float:
    """Give both AGVs dedicated left/right lanes within one pipeline."""
    lane_sign = -1.0 if agv_index == 0 else 1.0
    override = TASK_LANE_OVERRIDES.get((pipeline.index, task_index))
    if override is not None:
        lane_sign = override
    return pipeline.aisle_x + lane_sign * AISLE_LANE_OFFSET


def route_length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def canonical_segment_yaw(start, end) -> float:
    """Use forward/reverse differential drive with X/Y-aligned headings."""
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    return 0.0 if abs(delta[0]) >= abs(delta[1]) else math.pi / 2.0


def shortest_yaw_delta(start: float, end: float) -> float:
    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


def route_duration(
    points: list[tuple[float, float]],
    start_yaw: float = 0.0,
    hold_after_first: float = 0.0,
    start_hold: float = 0.0,
) -> float:
    duration = start_hold
    yaw = start_yaw
    for index, (start, end) in enumerate(zip(points, points[1:])):
        target_yaw = canonical_segment_yaw(start, end)
        duration += abs(shortest_yaw_delta(yaw, target_yaw)) / AGV_ANGULAR_SPEED
        duration += math.dist(start, end) / AGV_SPEED
        if index == 0:
            duration += hold_after_first
        yaw = target_yaw
    return duration


def sample_differential_route(
    points: list[tuple[float, float]],
    elapsed: float,
    start_yaw: float = 0.0,
    hold_after_first: float = 0.0,
    start_hold: float = 0.0,
):
    """Return position/yaw for rotate-in-place plus straight-line motion."""
    position = np.array(points[0], dtype=float)
    yaw = start_yaw
    elapsed = max(0.0, elapsed)
    if elapsed < start_hold:
        return position, yaw
    elapsed -= start_hold
    for index, (start, end) in enumerate(zip(points, points[1:])):
        target_yaw = canonical_segment_yaw(start, end)
        yaw_delta = shortest_yaw_delta(yaw, target_yaw)
        turn_time = abs(yaw_delta) / AGV_ANGULAR_SPEED
        if elapsed < turn_time:
            amount = elapsed / max(turn_time, 1e-9)
            return position, yaw + yaw_delta * smoothstep(amount)
        elapsed -= turn_time
        yaw = target_yaw
        distance = math.dist(start, end)
        travel_time = distance / AGV_SPEED
        if elapsed < travel_time:
            amount = elapsed / max(travel_time, 1e-9)
            return (
                (1.0 - amount) * np.array(start) + amount * np.array(end),
                yaw,
            )
        elapsed -= travel_time
        position = np.array(end, dtype=float)
        if index == 0 and hold_after_first > 0.0:
            if elapsed < hold_after_first:
                return position, yaw
            elapsed -= hold_after_first
    return np.array(points[-1], dtype=float), yaw


def sample_route(points: list[tuple[float, float]], amount: float):
    amount = float(np.clip(amount, 0.0, 1.0))
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    total = sum(lengths)
    if total <= 1e-9:
        return np.array(points[-1], dtype=float)
    target = amount * total
    traversed = 0.0
    for segment_index, (start, end, length) in enumerate(
        zip(points, points[1:], lengths)
    ):
        if target <= traversed + length or segment_index == len(lengths) - 1:
            local = np.clip((target - traversed) / max(length, 1e-9), 0.0, 1.0)
            return (1.0 - local) * np.array(start) + local * np.array(end)
        traversed += length
    return np.array(points[-1], dtype=float)


def sample_route_with_entry_hold(
    points: list[tuple[float, float]], elapsed: float, hold_duration: float
):
    """Compatibility position sampler for a differential-drive route."""
    return sample_differential_route(
        points, elapsed, start_yaw=0.0, hold_after_first=hold_duration
    )[0]


def interpolate_yaw(start: float, end: float, amount: float) -> float:
    delta = (end - start + math.pi) % (2.0 * math.pi) - math.pi
    return start + float(np.clip(amount, 0.0, 1.0)) * delta


def task_rack_yaw(item: ScheduledTask, now: float) -> float:
    if now <= item.pickup_departure:
        return item.task.pickup_rack_yaw
    if now < item.drop_arrival:
        amount = (now - item.pickup_departure) / max(
            item.drop_arrival - item.pickup_departure, 1e-9
        )
        return interpolate_yaw(
            item.task.pickup_rack_yaw,
            item.task.dropoff_rack_yaw,
            amount,
        )
    return item.task.dropoff_rack_yaw


def planned_agv_position(plan: Plan, agv_index: int, now: float) -> np.ndarray:
    """Sample the selected two-lane route for offline conflict validation."""
    position = np.array(plan.agv_start_positions[agv_index], dtype=float)
    for item in sorted(
        (value for value in plan.tasks if value.agv_index == agv_index),
        key=lambda value: value.start_time,
    ):
        if now < item.start_time:
            break
        if now < item.pickup_arrival:
            position = sample_differential_route(
                item.deadhead_route,
                now - item.start_time,
                start_hold=item.deadhead_hold_duration,
            )[0]
        elif now < item.pickup_departure:
            position = np.array(item.task.pickup)
        elif now < item.drop_arrival:
            position = sample_differential_route(
                item.loaded_route,
                now - item.pickup_departure,
                hold_after_first=item.loaded_hold_duration,
            )[0]
        elif now < item.drop_complete:
            position = np.array(item.task.dropoff)
        elif now < item.end_time:
            position = sample_differential_route(
                item.clearance_route, now - item.drop_complete
            )[0]
        else:
            position = np.array(item.clearance_route[-1])
        if now <= item.end_time:
            break
    return position


def planned_agv_yaw(plan: Plan, agv_index: int, now: float) -> float:
    yaw = 0.0
    for item in sorted(
        (value for value in plan.tasks if value.agv_index == agv_index),
        key=lambda value: value.start_time,
    ):
        if now < item.start_time:
            break
        if now < item.pickup_arrival:
            yaw = sample_differential_route(
                item.deadhead_route,
                now - item.start_time,
                start_hold=item.deadhead_hold_duration,
            )[1]
        elif now < item.pickup_departure:
            yaw = sample_differential_route(item.deadhead_route, float("inf"))[1]
        elif now < item.drop_arrival:
            yaw = sample_differential_route(
                item.loaded_route,
                now - item.pickup_departure,
                hold_after_first=item.loaded_hold_duration,
            )[1]
        elif now < item.drop_complete:
            yaw = sample_differential_route(item.loaded_route, float("inf"))[1]
        elif now < item.end_time:
            yaw = sample_differential_route(
                item.clearance_route, now - item.drop_complete
            )[1]
        else:
            yaw = sample_differential_route(
                item.clearance_route, float("inf")
            )[1]
        if now <= item.end_time:
            break
    return yaw


def active_scheduled_task(plan: Plan, agv_index: int, now: float):
    for item in sorted(
        (value for value in plan.tasks if value.agv_index == agv_index),
        key=lambda value: value.start_time,
    ):
        if item.start_time <= now <= item.end_time:
            return item
    return None


def rotated_axis_extents(extents: np.ndarray, yaw: float) -> np.ndarray:
    cosine = abs(math.cos(yaw))
    sine = abs(math.sin(yaw))
    return np.array(
        [cosine * extents[0] + sine * extents[1],
         sine * extents[0] + cosine * extents[1]],
        dtype=float,
    )


def vehicle_half_extents(
    item: ScheduledTask | None, now: float, yaw: float
) -> np.ndarray:
    extents = rotated_axis_extents(AGV_HALF_EXTENTS, yaw)
    if item is not None and item.pickup_arrival <= now <= item.drop_complete:
        rack_extents = rotated_axis_extents(
            RACK_HALF_EXTENTS, yaw
        )
        extents = np.maximum(extents, rack_extents)
    return extents + TRAFFIC_SAFETY_MARGIN


def pair_clearance_factor(
    first_plan: Plan,
    first_agv: int,
    second_plan: Plan,
    second_agv: int,
    now: float,
) -> float:
    first_position = planned_agv_position(first_plan, first_agv, now)
    second_position = planned_agv_position(second_plan, second_agv, now)
    first_item = active_scheduled_task(first_plan, first_agv, now)
    second_item = active_scheduled_task(second_plan, second_agv, now)
    first_yaw = planned_agv_yaw(first_plan, first_agv, now)
    second_yaw = planned_agv_yaw(second_plan, second_agv, now)
    required = vehicle_half_extents(first_item, now, first_yaw) + vehicle_half_extents(
        second_item, now, second_yaw
    )
    separation = np.abs(first_position - second_position)
    # Axis-aligned AGVs are separated as soon as either axis clears its summed
    # half extent.  Values below one mean their safety rectangles overlap.
    return float(np.max(separation / required))


def first_traffic_conflict(plans: list[Plan], time_step: float = 0.05):
    vehicles = [(plan, agv) for plan in plans for agv in (0, 1)]
    horizon = max(plan.makespan for plan in plans)
    for now in np.arange(0.0, horizon + time_step, time_step):
        for first_index in range(len(vehicles)):
            for second_index in range(first_index + 1, len(vehicles)):
                first = vehicles[first_index]
                second = vehicles[second_index]
                factor = pair_clearance_factor(*first, *second, now)
                if factor < 1.0:
                    return "Factory", now, first, second, factor
    return None


def validate_agv_clearance(plans: list[Plan]):
    """Check all eight AGVs, including turns into central parking."""
    results = {}
    vehicles = [(plan, agv) for plan in plans for agv in (0, 1)]
    smallest_factor = float("inf")
    smallest_distance = float("inf")
    horizon = max(plan.makespan for plan in plans)
    for now in np.arange(0.0, horizon + 0.05, 0.05):
        for first_index in range(len(vehicles)):
            for second_index in range(first_index + 1, len(vehicles)):
                first = vehicles[first_index]
                second = vehicles[second_index]
                first_position = planned_agv_position(*first, now)
                second_position = planned_agv_position(*second, now)
                smallest_distance = min(
                    smallest_distance,
                    float(np.linalg.norm(first_position - second_position)),
                )
                smallest_factor = min(
                    smallest_factor,
                    pair_clearance_factor(*first, *second, now),
                )
    if smallest_factor < 1.0:
        raise RuntimeError(
            "AGV cooperative reservation failed in factory: "
            f"clearance factor {smallest_factor:.3f}"
        )
    results["Factory"] = (smallest_factor, smallest_distance)
    return results


def build_tasks(pipeline: PipelineSpec, stock_rack_path: str) -> list[Task]:
    poses = rack_pose_map()
    p1, p2, p3 = pipeline.pairs
    raw = {pair: poses[(pipeline.bank, pair, "RawMaterial")] for pair in pipeline.pairs}
    finished = {
        pair: poses[(pipeline.bank, pair, "FinishedProduct")]
        for pair in pipeline.pairs
    }
    return [
        # Phase 1: Cell3 final product leaves while its empty raw rack becomes
        # the next empty finished rack.
        Task(
            0,
            f"L{pipeline.index} Cell{p3} product -> finished zone",
            finished[p3],
            pipeline.finished_stock_xy,
            rack_path(pipeline.bank, p3, False),
        ),
        Task(
            1,
            f"L{pipeline.index} Cell{p3} empty raw -> Cell{p3} finished",
            raw[p3],
            finished[p3],
            rack_path(pipeline.bank, p3, True),
            vacancy_dependency=0,
        ),
        # Phase 2: Cell2 output becomes Cell3 input; Cell2's empty input rack
        # moves to its output station.
        Task(
            2,
            f"L{pipeline.index} Cell{p2} product -> Cell{p3} raw",
            finished[p2],
            raw[p3],
            rack_path(pipeline.bank, p2, False),
            dependencies=(0, 1),
            trigger_pair=p3,
        ),
        Task(
            3,
            f"L{pipeline.index} Cell{p2} empty raw -> Cell{p2} finished",
            raw[p2],
            finished[p2],
            rack_path(pipeline.bank, p2, True),
            dependencies=(0, 1),
            vacancy_dependency=2,
        ),
        # Phase 3: Cell1 output becomes Cell2 input.  Cell1's empty raw rack
        # must first occupy its finished station before a full stock rack can
        # be delivered to the now-vacant raw station.
        Task(
            4,
            f"L{pipeline.index} Cell{p1} product -> Cell{p2} raw",
            finished[p1],
            raw[p2],
            rack_path(pipeline.bank, p1, False),
            dependencies=(2, 3),
            trigger_pair=p2,
        ),
        Task(
            5,
            f"L{pipeline.index} Cell{p1} empty raw -> Cell{p1} finished",
            raw[p1],
            finished[p1],
            rack_path(pipeline.bank, p1, True),
            dependencies=(2, 3),
            vacancy_dependency=4,
        ),
        Task(
            6,
            f"L{pipeline.index} raw stock -> Cell{p1} raw",
            pipeline.raw_stock_xy,
            raw[p1],
            stock_rack_path,
            dependencies=(5,),
            trigger_pair=p1,
        ),
    ]


def evaluate_assignment(
    pipeline: PipelineSpec,
    tasks: list[Task],
    assignment: tuple[int, ...],
    starts: tuple[tuple[float, float], tuple[float, float]],
    traffic_holds: dict[int, float] | None = None,
    deadhead_holds: dict[int, float] | None = None,
) -> tuple[list[ScheduledTask], float, float]:
    traffic_holds = traffic_holds or {}
    deadhead_holds = deadhead_holds or {}
    available = [
        pipeline.start_delay,
        pipeline.start_delay + AGV2_INITIAL_STAGGER,
    ]
    positions = [starts[0], starts[1]]
    finish_times: dict[int, float] = {}
    pickup_departure_times: dict[int, float] = {}
    pickup_clear_times: dict[int, float] = {}
    scheduled = []
    total_distance = 0.0
    last_task_index_by_agv = {
        agv_index: max(
            task.index
            for task, assigned_agv in zip(tasks, assignment)
            if assigned_agv == agv_index
        )
        for agv_index in (0, 1)
    }
    for task, agv_index in zip(tasks, assignment):
        deadhead_lane_x = traffic_lane_x(
            pipeline,
            positions[agv_index],
            task.pickup,
            agv_index,
            None,
        )
        loaded_lane_x = traffic_lane_x(
            pipeline, task.pickup, task.dropoff, agv_index, task.index
        )
        deadhead = make_route(
            positions[agv_index],
            task.pickup,
            deadhead_lane_x,
        )
        loaded = make_route(
            task.pickup,
            task.dropoff,
            loaded_lane_x,
        )
        deadhead_distance = route_length(deadhead)
        loaded_distance = route_length(loaded)
        is_final_agv_task = task.index == last_task_index_by_agv[agv_index]
        clearance_lane_x = traffic_lane_x(
            pipeline, task.dropoff, task.dropoff, agv_index, None
        )
        clearance = [task.dropoff]
        clearance_target = (clearance_lane_x, task.dropoff[1])
        if math.dist(task.dropoff, clearance_target) > 1e-6:
            clearance.append(clearance_target)
        station_clearance_distance = route_length(clearance)
        if is_final_agv_task:
            parking_target = pipeline.parking_xy[agv_index]
            # Leave the production aisle through the transverse connector at
            # y=0, then turn into the central vertical parking corridor.
            for point in (
                (clearance_lane_x, 0.0),
                (parking_target[0], 0.0),
                parking_target,
            ):
                if math.dist(clearance[-1], point) > 1e-6:
                    clearance.append(point)
        clearance_distance = route_length(clearance)
        release = max((finish_times[index] for index in task.dependencies), default=0.0)
        start_time = max(available[agv_index], release)
        loaded_hold_duration = traffic_holds.get(task.index, 0.0)
        deadhead_hold_duration = deadhead_holds.get(task.index, 0.0)
        deadhead_duration = route_duration(
            deadhead, start_hold=deadhead_hold_duration
        )
        loaded_duration = route_duration(
            loaded, hold_after_first=loaded_hold_duration
        )
        clearance_duration = route_duration(clearance)
        pickup_arrival = start_time + deadhead_duration
        pickup_departure = pickup_arrival + PICKUP_DWELL
        drop_arrival = pickup_departure + loaded_duration
        # A vehicle may leave early and approach an occupied destination; only
        # its arrival must follow the preceding pickup that vacates the stand.
        if task.vacancy_dependency is not None:
            vacancy_release = pickup_clear_times[task.vacancy_dependency]
            drop_approach_distance = (
                math.dist(loaded[-2], loaded[-1]) if len(loaded) > 1 else 0.0
            )
            earliest_drop_arrival = (
                vacancy_release
                + STATION_MOUTH_CLEARANCE
                + (math.pi / 2.0) / AGV_ANGULAR_SPEED
                + drop_approach_distance / AGV_SPEED
            )
            if drop_arrival < earliest_drop_arrival:
                wait = earliest_drop_arrival - drop_arrival
                start_time += wait
                pickup_arrival += wait
                pickup_departure += wait
                drop_arrival += wait
        drop_complete = drop_arrival + DROP_DWELL
        station_clear_time = drop_complete + route_duration(clearance[:2])
        end_time = drop_complete + clearance_duration
        scheduled.append(
            ScheduledTask(
                task=task,
                agv_index=agv_index,
                start_position=positions[agv_index],
                start_time=start_time,
                pickup_arrival=pickup_arrival,
                pickup_departure=pickup_departure,
                drop_arrival=drop_arrival,
                drop_complete=drop_complete,
                station_clear_time=station_clear_time,
                end_time=end_time,
                deadhead_route=deadhead,
                loaded_route=loaded,
                clearance_route=clearance,
                travel_distance=(
                    deadhead_distance + loaded_distance + clearance_distance
                ),
                loaded_hold_duration=loaded_hold_duration,
                deadhead_hold_duration=deadhead_hold_duration,
            )
        )
        available[agv_index] = end_time
        positions[agv_index] = clearance[-1]
        finish_times[task.index] = end_time
        pickup_departure_times[task.index] = pickup_departure
        pickup_exit_distance = (
            math.dist(loaded[0], loaded[1]) if len(loaded) > 1 else 0.0
        )
        pickup_clear_times[task.index] = (
            pickup_departure + route_duration(loaded[:2])
        )
        total_distance += deadhead_distance + loaded_distance + clearance_distance
    return scheduled, max(available), total_distance


def dynamic_program_line(
    pipeline: PipelineSpec,
    stock_rack_path: str,
    traffic_holds: dict[int, float] | None = None,
    deadhead_holds: dict[int, float] | None = None,
) -> Plan:
    tasks = build_tasks(pipeline, stock_rack_path)
    cell_y = [rack_pose_map()[(pipeline.bank, pair, "RawMaterial")][1] for pair in pipeline.pairs]
    center_y = float(np.mean(cell_y))
    starts = (
        (
            traffic_lane_x(pipeline, (0.0, 0.0), (0.0, 0.0), 0, None),
            center_y - 0.75,
        ),
        (
            traffic_lane_x(pipeline, (0.0, 0.0), (0.0, 0.0), 1, None),
            center_y + 0.75,
        ),
    )
    # The user-specified takt fixes the assignment.  Dynamic route planning is
    # still applied to each leg through the two direction-dependent aisle lanes.
    assignment = (0, 1, 0, 1, 0, 1, 1)
    scheduled, makespan, distance = evaluate_assignment(
        pipeline, tasks, assignment, starts, traffic_holds, deadhead_holds
    )
    triggers = {
        item.task.trigger_pair: item.station_clear_time + ROBOT_CLEARANCE_DELAY
        for item in scheduled
        if item.task.trigger_pair is not None
    }
    return Plan(
        pipeline=pipeline,
        tasks=scheduled,
        agv_start_positions=starts,
        robot_trigger_times=triggers,
        makespan=makespan,
        total_distance=distance,
    )


def sample_keyframes(frames, elapsed: float):
    if elapsed <= 0.0:
        frame = frames[0]
        return frame.base, frame.active, frame.finger_opening, frame.machine_1_press_z, frame.machine_2_press_z, frame.tube_state, False
    traversed = 0.0
    for previous, current in zip(frames, frames[1:]):
        if elapsed <= traversed + current.duration:
            linear = (elapsed - traversed) / current.duration
            amount = smoothstep(linear)
            return (
                interpolate_base(previous.base, current.base, amount),
                (1.0 - amount) * previous.active + amount * current.active,
                (1.0 - amount) * previous.finger_opening + amount * current.finger_opening,
                (1.0 - amount) * previous.machine_1_press_z + amount * current.machine_1_press_z,
                (1.0 - amount) * previous.machine_2_press_z + amount * current.machine_2_press_z,
                current.tube_state,
                True,
            )
        traversed += current.duration
    frame = frames[-1]
    return frame.base, frame.active, frame.finger_opening, frame.machine_1_press_z, frame.machine_2_press_z, frame.tube_state, False


def sample_continuous_process(cycle_frames, elapsed: float):
    """Select and sample one of the 48 persistent two-tube work cycles."""
    elapsed = max(0.0, elapsed)
    for cycle_index, frames in enumerate(cycle_frames):
        duration = sum(frame.duration for frame in frames[1:])
        if elapsed <= duration or cycle_index == len(cycle_frames) - 1:
            return cycle_index, sample_keyframes(frames, min(elapsed, duration))
        elapsed -= duration
    raise RuntimeError("Continuous robot process has no cycles")


def main() -> None:
    args = parse_args()
    kin, active_names, cycle_frames = solve_all_cycles()
    robot_process_duration = sum(
        frame.duration
        for frames in cycle_frames
        for frame in frames[1:]
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
        from pxr import Gf, Sdf, UsdGeom

        if not open_stage(str(SCENE_PATH)):
            raise RuntimeError(f"Could not open {SCENE_PATH}")
        while is_stage_loading():
            app.update()
        for _ in range(5):
            app.update()
        stage = omni.usd.get_context().get_stage()
        root_layer = stage.GetRootLayer()
        UsdGeom.Xform.Define(stage, DEMO_ROOT)
        UsdGeom.Xform.Define(stage, f"{DEMO_ROOT}/StockRacks")
        UsdGeom.Xform.Define(stage, f"{DEMO_ROOT}/AGVs")
        UsdGeom.Xform.Define(stage, f"{DEMO_ROOT}/Routes")
        UsdGeom.Xform.Define(stage, f"{DEMO_ROOT}/ProcessedTubes")

        pipelines = pipeline_specs()
        # Copy four active and four spare full raw racks into the north stock zone
        # before hiding the initially empty raw racks at the cells.
        stock_rack_paths = {}

        def place_copied_rack(prim, xy, yaw=RACK_WORLD_YAW):
            xform = UsdGeom.Xformable(prim)
            translate = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))
            rotate = UsdGeom.XformOp(prim.GetAttribute("xformOp:rotateZ"))
            xform.SetXformOpOrder([translate, rotate])
            translate.Set(Gf.Vec3d(xy[0], xy[1], 0.0))
            rotate.Set(math.degrees(yaw))

        for pipeline in pipelines:
            source = rack_path(pipeline.bank, pipeline.pairs[0], True)
            destination = f"{DEMO_ROOT}/StockRacks/Line_{pipeline.index:02d}_RawSupply"
            Sdf.CopySpec(root_layer, source, root_layer, destination)
            if not stage.GetPrimAtPath(destination):
                raise RuntimeError(f"Could not copy stock rack from {source}")
            stock_rack_paths[pipeline.index] = destination
            prim = stage.GetPrimAtPath(destination)
            place_copied_rack(prim, pipeline.raw_stock_xy)

            # Two outside-accessible columns x ten rows fill the raw stock
            # area.  No AGV has to cross another rack to reach its target.
            for spare_row in range(1, 5):
                spare = (
                    f"{DEMO_ROOT}/StockRacks/Line_{pipeline.index:02d}_"
                    f"Spare_{spare_row:02d}"
                )
                Sdf.CopySpec(root_layer, source, root_layer, spare)
                if not stage.GetPrimAtPath(spare):
                    raise RuntimeError(f"Could not copy spare stock rack from {source}")
                place_copied_rack(
                    stage.GetPrimAtPath(spare),
                    (
                        pipeline.raw_stock_xy[0],
                        pipeline.raw_stock_xy[1] + spare_row * 1.0,
                    ),
                    yaw=RACK_WORLD_YAW,
                )

        # Initial condition: every cell raw rack is empty and every finished
        # rack contains the two completed 48-tube bins.
        for pipeline in pipelines:
            for pair in pipeline.pairs:
                raw_root = rack_path(pipeline.bank, pair, True)
                finished_root = rack_path(pipeline.bank, pair, False)
                for bin_index in (1, 2):
                    raw_tubes = (
                        f"{raw_root}/Bins/Bin_{bin_index:02d}/WorldYawAlignment/"
                        "AisleAxisTilt/Orientation/Tubes"
                    )
                    finished_tubes = (
                        f"{finished_root}/Bins/Bin_{bin_index:02d}/WorldYawAlignment/"
                        "AisleAxisTilt/Orientation/Tubes"
                    )
                    if not stage.GetPrimAtPath(finished_tubes):
                        Sdf.CopySpec(root_layer, raw_tubes, root_layer, finished_tubes)
                        if not stage.GetPrimAtPath(finished_tubes):
                            raise RuntimeError(f"Could not fill finished bin: {finished_tubes}")
                    UsdGeom.Imageable(stage.GetPrimAtPath(raw_tubes)).MakeInvisible()

        plans = [
            dynamic_program_line(
                pipeline,
                stock_rack_paths[pipeline.index],
                COOPERATIVE_ENTRY_HOLDS.get(pipeline.index),
                COOPERATIVE_DEADHEAD_HOLDS.get(pipeline.index),
            )
            for pipeline in pipelines
        ]
        clearance_results = validate_agv_clearance(plans)
        for bank, (clearance_factor, center_distance) in clearance_results.items():
            print(
                f"V2V RESERVATION {bank}: clearance factor "
                f"{clearance_factor:.3f}, minimum center distance "
                f"{center_distance:.3f} m",
                flush=True,
            )
        for plan in plans:
            print(
                f"DP LINE {plan.pipeline.index}: makespan={plan.makespan:.2f}s "
                f"distance={plan.total_distance:.2f}m",
                flush=True,
            )
            for item in sorted(plan.tasks, key=lambda value: value.start_time):
                print(
                    f"  t={item.start_time:5.1f}-{item.end_time:5.1f} "
                    f"AGV-{item.agv_index + 1}: {item.task.name} "
                    f"[pre-entry hold={item.deadhead_hold_duration:.2f}s, "
                    f"loaded-entry hold={item.loaded_hold_duration:.2f}s]",
                    flush=True,
                )
                if (plan.pipeline.index, item.task.index) in TASK_LANE_OVERRIDES:
                    print(
                        f"    V2V LANE CHANGE: line {plan.pipeline.index} "
                        f"AGV-{item.agv_index + 1} uses alternate lane for task "
                        f"{item.task.index}",
                        flush=True,
                    )

        # Create a kinematic visual hierarchy from the exact two meshes used by
        # HIK-Q2-400D.urdf.  Keeping rigid-body/articulation APIs out of this
        # scheduling preview prevents all referenced roots from being solved at
        # the same physics origin and exploding apart.  The vehicle root is the
        # mesh XY center; LiftPlatform owns the real 60 mm vertical motion.
        agv_visuals = {}
        agv_body_material = create_material(
            stage,
            f"{DEMO_ROOT}/Materials/AGVBody",
            (0.10, 0.15, 0.20, 1.0),
        )
        agv_lift_materials = (
            create_material(
                stage,
                f"{DEMO_ROOT}/Materials/AGV1TopRed",
                (0.90, 0.04, 0.03, 1.0),
            ),
            create_material(
                stage,
                f"{DEMO_ROOT}/Materials/AGV2TopYellow",
                (0.98, 0.78, 0.04, 1.0),
            ),
        )
        route_colors = ((0.05, 0.85, 1.0), (1.0, 0.18, 0.68))
        for plan in plans:
            for agv_index in (0, 1):
                path = f"{DEMO_ROOT}/AGVs/Line_{plan.pipeline.index:02d}/AGV_{agv_index + 1}"
                xform = UsdGeom.Xform.Define(stage, path)
                xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                # At cell racks yaw=0: the AGV short local-Y edge enters the
                # rack's long world-Y opening.  Stock pickup rotates to yaw=90.
                xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(0.0)
                add_stl_mesh(
                    stage,
                    f"{path}/BaseLink/Geometry",
                    AGV_MESH_DIR / "base_link.STL",
                    agv_body_material,
                    collision=False,
                )
                lift = UsdGeom.Xform.Define(stage, f"{path}/LiftPlatform")
                lift_op = lift.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                add_stl_mesh(
                    stage,
                    f"{path}/LiftPlatform/Geometry",
                    AGV_MESH_DIR / "box.STL",
                    agv_lift_materials[agv_index],
                    collision=False,
                )
                agv_visuals[(plan.pipeline.index, agv_index)] = (xform, lift_op)
            for item in plan.tasks:
                points = (
                    item.deadhead_route
                    + item.loaded_route[1:]
                    + item.clearance_route[1:]
                )
                curve = UsdGeom.BasisCurves.Define(
                    stage,
                    f"{DEMO_ROOT}/Routes/Line_{plan.pipeline.index:02d}_Task_{item.task.index:02d}",
                )
                curve.CreateTypeAttr("linear")
                curve.CreateCurveVertexCountsAttr([len(points)])
                curve.CreatePointsAttr(
                    [Gf.Vec3f(point[0], point[1], 0.018) for point in points]
                )
                curve.CreateWidthsAttr([0.035] * len(points))
                curve.CreateDisplayColorAttr([Gf.Vec3f(*route_colors[item.agv_index])])

        cell_specs = build_cell_specs()
        incoming_rack_paths = {
            (plan.pipeline.bank, item.task.trigger_pair): item.task.rack_path
            for plan in plans
            for item in plan.tasks
            if item.task.trigger_pair is not None
        }
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / args.fps,
            rendering_dt=1.0 / args.fps,
        )
        robot_runtimes = {}
        for spec in cell_specs:
            robot = world.scene.add(
                SingleArticulation(
                    prim_path=spec.robot_path,
                    name=f"pipeline_robot_{spec.index:02d}",
                )
            )
            press_ops = {}
            for name, path in (
                ("machine_1", spec.machine_1_press_path),
                ("machine_2", spec.machine_2_press_path),
            ):
                press_ops[name] = UsdGeom.Xformable(stage.GetPrimAtPath(path)).AddTranslateOp(
                    UsdGeom.XformOp.PrecisionDouble,
                    opSuffix="agvPipelineDemo",
                )
            source_sets = []
            carried_sets = []
            translate_sets = []
            orient_sets = []
            incoming_root = incoming_rack_paths[(spec.bank, spec.pair)]
            for cycle_number, pickup in enumerate(
                CONTINUOUS_PICKUP_SCHEDULE, start=1
            ):
                sources = {}
                carried = {}
                translates = {}
                orients = {}
                for side in ("left", "right"):
                    column_name = pickup["paths"][side].rsplit("/", 1)[-1]
                    source_path = (
                        f"{incoming_root}/Bins/Bin_{pickup['bin_index']:02d}/"
                        "WorldYawAlignment/AisleAxisTilt/Orientation/Tubes/"
                        f"{column_name}"
                    )
                    source_prim = stage.GetPrimAtPath(source_path)
                    if not source_prim:
                        raise RuntimeError(f"Missing incoming tube: {source_path}")
                    sources[side] = UsdGeom.Imageable(source_prim)
                    carried_path = (
                        f"{DEMO_ROOT}/ProcessedTubes/{spec.key}/"
                        f"Cycle_{cycle_number:02d}/Tube_{side.capitalize()}"
                    )
                    carried_xform = UsdGeom.Xform.Define(stage, carried_path)
                    carried_xform.GetPrim().GetReferences().AddReference(
                        str(TUBE_ASSET_PATH)
                    )
                    carried[side] = UsdGeom.Imageable(carried_xform.GetPrim())
                    carried[side].MakeInvisible()
                    translates[side] = carried_xform.AddTranslateOp(
                        UsdGeom.XformOp.PrecisionDouble
                    )
                    orients[side] = carried_xform.AddOrientOp(
                        UsdGeom.XformOp.PrecisionDouble
                    )
                source_sets.append(sources)
                carried_sets.append(carried)
                translate_sets.append(translates)
                orient_sets.append(orients)
            robot_runtimes[(spec.bank, spec.pair)] = {
                "spec": spec,
                "robot": robot,
                "presses": press_ops,
                "sources": source_sets,
                "carried": carried_sets,
                "translates": translate_sets,
                "orients": orient_sets,
                "placed": [
                    {"left": None, "right": None} for _ in cycle_frames
                ],
            }

        world.reset()
        for runtime in robot_runtimes.values():
            robot = runtime["robot"]
            if not robot.handles_initialized:
                raise RuntimeError(f"Robot articulation failed: {robot.prim_path}")
            robot.disable_gravity()

        trigger_times = {}
        for plan in plans:
            for pair, trigger in plan.robot_trigger_times.items():
                trigger_times[(plan.pipeline.bank, pair)] = trigger

        def set_root_pose(
            prim_path: str,
            xy,
            z: float,
            yaw: float,
            visible: bool = True,
        ):
            prim = stage.GetPrimAtPath(prim_path)
            UsdGeom.Imageable(prim).MakeVisible() if visible else UsdGeom.Imageable(prim).MakeInvisible()
            xform = UsdGeom.Xformable(prim)
            ordered = xform.GetOrderedXformOps()
            translate = next((op for op in ordered if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
            rotate = next((op for op in ordered if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ), None)
            if translate is None:
                translate = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            if rotate is None:
                rotate = xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
            translate.Set(Gf.Vec3d(float(xy[0]), float(xy[1]), z))
            rotate.Set(math.degrees(yaw))

        def sample_task_pose(item: ScheduledTask, now: float):
            if now < item.start_time:
                return np.array(item.start_position), 0.0
            if now < item.pickup_arrival:
                return sample_differential_route(
                    item.deadhead_route,
                    now - item.start_time,
                    start_hold=item.deadhead_hold_duration,
                )
            if now < item.pickup_departure:
                return sample_differential_route(item.deadhead_route, float("inf"))
            if now < item.drop_arrival:
                return sample_differential_route(
                    item.loaded_route,
                    now - item.pickup_departure,
                    hold_after_first=item.loaded_hold_duration,
                )
            if now < item.drop_complete:
                return sample_differential_route(item.loaded_route, float("inf"))
            if now < item.end_time:
                return sample_differential_route(
                    item.clearance_route, now - item.drop_complete
                )
            return sample_differential_route(item.clearance_route, float("inf"))

        def sample_task_position(item: ScheduledTask, now: float):
            return sample_task_pose(item, now)[0]

        if not args.headless:
            try:
                from isaacsim.core.utils.viewports import set_camera_view

                set_camera_view(
                    eye=np.array([21.0, -31.0, 24.0]),
                    target=np.array([0.0, 0.0, 0.6]),
                    camera_prim_path="/World/Camera",
                )
            except Exception as exc:
                print(f"Camera setup skipped: {exc}", flush=True)

        latest_robot_completion = max(
            trigger + robot_process_duration * ROBOT_TIME_SCALE
            for trigger in trigger_times.values()
        )
        demo_duration = max(
            max(plan.makespan for plan in plans),
            latest_robot_completion,
        ) + 4.0
        if args.check:
            sample_times = [0.0, demo_duration * 0.35, demo_duration * 0.70, demo_duration]
        else:
            sample_times = None
        started_at = time.perf_counter()
        last_announced = set()
        while app.is_running():
            now = (
                sample_times.pop(0)
                if sample_times is not None and sample_times
                else time.perf_counter() - started_at
            )

            for plan in plans:
                for agv_index in (0, 1):
                    assigned = sorted(
                        (item for item in plan.tasks if item.agv_index == agv_index),
                        key=lambda value: value.start_time,
                    )
                    position = np.array(plan.agv_start_positions[agv_index])
                    lift_height = 0.0
                    agv_yaw = 0.0
                    for item in assigned:
                        if now < item.start_time:
                            break
                        position, agv_yaw = sample_task_pose(item, now)
                        if now <= item.end_time:
                            if item.pickup_arrival <= now < item.pickup_departure:
                                lift_height = AGV_LIFT_TRAVEL * smoothstep(
                                    (now - item.pickup_arrival)
                                    / max(item.pickup_departure - item.pickup_arrival, 1e-9)
                                )
                            elif item.pickup_departure <= now < item.drop_arrival:
                                lift_height = AGV_LIFT_TRAVEL
                            elif item.drop_arrival <= now < item.drop_complete:
                                lift_height = AGV_LIFT_TRAVEL * (
                                    1.0
                                    - smoothstep(
                                        (now - item.drop_arrival)
                                        / max(item.drop_complete - item.drop_arrival, 1e-9)
                                    )
                                )
                            break
                        position = np.array(item.clearance_route[-1])
                    agv_root, lift_op = agv_visuals[(plan.pipeline.index, agv_index)]
                    agv_root.GetOrderedXformOps()[0].Set(
                        Gf.Vec3d(float(position[0]), float(position[1]), 0.0)
                    )
                    agv_root.GetOrderedXformOps()[1].Set(math.degrees(agv_yaw))
                    lift_op.Set(Gf.Vec3d(0.0, 0.0, lift_height))

                for item in plan.tasks:
                    if now < item.pickup_arrival:
                        set_root_pose(
                            item.task.rack_path,
                            item.task.pickup,
                            0.0,
                            item.task.pickup_rack_yaw,
                        )
                    elif now < item.pickup_departure:
                        lift = smoothstep(
                            (now - item.pickup_arrival)
                            / max(item.pickup_departure - item.pickup_arrival, 1e-9)
                        )
                        set_root_pose(
                            item.task.rack_path,
                            item.task.pickup,
                            AGV_LIFT_TRAVEL * lift,
                            item.task.pickup_rack_yaw,
                        )
                    elif now < item.drop_arrival:
                        position, carrier_yaw = sample_task_pose(item, now)
                        set_root_pose(
                            item.task.rack_path,
                            position,
                            AGV_LIFT_TRAVEL,
                            carrier_yaw + math.pi / 2.0,
                        )
                    elif now < item.drop_complete:
                        lower = smoothstep(
                            (now - item.drop_arrival)
                            / max(item.drop_complete - item.drop_arrival, 1e-9)
                        )
                        set_root_pose(
                            item.task.rack_path,
                            item.task.dropoff,
                            AGV_LIFT_TRAVEL * (1.0 - lower),
                            item.task.dropoff_rack_yaw,
                        )
                    else:
                        set_root_pose(
                            item.task.rack_path,
                            item.task.dropoff,
                            0.0,
                            item.task.dropoff_rack_yaw,
                            visible=not item.task.disappear,
                        )

            for key, runtime in robot_runtimes.items():
                spec = runtime["spec"]
                robot = runtime["robot"]
                presses = runtime["presses"]
                trigger = trigger_times[key]
                process_elapsed = max(0.0, now - trigger) / ROBOT_TIME_SCALE
                cycle_index, sampled = sample_continuous_process(
                    cycle_frames, process_elapsed
                )
                base, active, finger, press_1, press_2, tube_state, active_now = sampled
                transformed_base = transform_base(base, spec)
                transformed_active = mirror_active(active_names, active) if spec.east else active
                quaternion = np.array(
                    [math.cos(transformed_base[2] / 2.0), 0.0, 0.0, math.sin(transformed_base[2] / 2.0)],
                    dtype=np.float32,
                )
                robot.set_world_pose(
                    position=np.array([transformed_base[0], transformed_base[1], BASE_GROUND_Z]),
                    orientation=quaternion,
                )
                robot.set_joint_positions(
                    full_joint_vector(robot.dof_names, active_names, transformed_active, finger)
                )
                robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))
                robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
                robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
                presses["machine_1"].Set(Gf.Vec3d(0.0, 0.0, press_1))
                presses["machine_2"].Set(Gf.Vec3d(0.0, 0.0, press_2))

                # The actual incoming rack supplies each pair.  Once picked,
                # the source tube stays absent and its carried copy remains in
                # the corresponding finished-bin hole after release.
                sources = runtime["sources"][cycle_index]
                carried = runtime["carried"][cycle_index]
                translates = runtime["translates"][cycle_index]
                orients = runtime["orients"][cycle_index]
                placed = runtime["placed"][cycle_index]
                for side in ("left", "right"):
                    _, reference_position, reference_matrix = (
                        finger_and_tube_transform(
                            kin, active_names, base, active, side
                        )
                    )
                    if tube_state == "source":
                        sources[side].MakeVisible()
                        carried[side].MakeInvisible()
                        continue
                    sources[side].MakeInvisible()
                    carried[side].MakeVisible()
                    current = transform_tube_pose(
                        reference_position, reference_matrix, spec
                    )
                    if tube_state == "attached":
                        placed[side] = None
                    elif placed[side] is None:
                        placed[side] = current
                    if tube_state == "placed":
                        current = placed[side]
                    position, matrix = current
                    quat_xyzw = Rotation.from_matrix(matrix).as_quat()
                    translates[side].Set(Gf.Vec3d(*position))
                    orients[side].Set(
                        Gf.Quatd(quat_xyzw[3], Gf.Vec3d(*quat_xyzw[:3]))
                    )
                if now >= trigger and active_now and key not in last_announced:
                    print(
                        f"ROBOT START {key[0]} Cell {key[1]} at t={now:.1f}s: "
                        "new raw rack arrived and both AGVs cleared",
                        flush=True,
                    )
                    last_announced.add(key)

            world.step(render=not args.headless)
            if sample_times is not None:
                if not sample_times:
                    print("CHECK PASSED: AGV DP schedule, rack transport, and robot triggers", flush=True)
                    return
            elif now >= demo_duration:
                print(
                    "AGV DEMO COMPLETE: empty racks exited, intermediate products advanced, "
                    "final products stored, and new raw racks triggered robots",
                    flush=True,
                )
                while app.is_running():
                    world.step(render=True)
                return
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
