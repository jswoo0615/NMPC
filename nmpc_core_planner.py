#!/usr/bin/env python

import math
from enum import Enum
from collections import deque
import random
import numpy as np

import carla
from agents.tools.misc import distance_vehicle, draw_waypoints
import nmpc_core

class RoadOption(Enum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4

class LocalPlanner(object):
    HORIZON_STEPS = 100
    MAX_WP_BUFFER = 200

    def __init__(self, vehicle, opt_dict=None):
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()
        
        self._target_speed = 5.0        # m/s
        self._sampling_radius = 1.0

        self._waypoints_queue = deque(maxlen=20000)
        self._waypoint_buffer = deque(maxlen=self.MAX_WP_BUFFER)
        self._global_plan = False

        self._ego_state = np.zeros(6, dtype=np.float64)
        self._wp_x = np.zeros(self.MAX_WP_BUFFER, dtype=np.float64)
        self._wp_y = np.zeros(self.MAX_WP_BUFFER, dtype=np.float64)
        self._obstacles = np.zeros((10, 5), dtype=np.float64)

        for i in range(10):
            self._obstacles[i, 0] = -1000.0
            self._obstacles[i, 1] = 0.0
            self._obstacles[i, 2] = 0.1

        self._nmpc_solver = nmpc_core.SparseNMPCWrapper(
            self._ego_state, self._wp_x, self._wp_y, self._obstacles
        )

        if opt_dict and 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed'] / 3.6

        self._nmpc_solver.set_target_speed(self._target_speed)

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))
        self._compute_next_waypoints(k=200)

    def set_speed(self, speed):
        self._target_speed = speed / 3.6
        self._nmpc_solver.set_target_speed(self._target_speed)

    def _compute_next_waypoints(self, k=1):
        available = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        for _ in range(min(available, k)):
            last_wp = self._waypoints_queue[-1][0]
            next_wps = list(last_wp.next(self._sampling_radius))
            next_wp = next_wps[0] if len(next_wps) == 1 else random.choice(next_wps)
            self._waypoints_queue.append((next_wp, RoadOption.LANEFOLLOW))
    
    def set_global_plan(self, current_plan):
        self._waypoints_queue.clear()
        for elem in current_plan:
            if isinstance(elem, tuple):
                self._waypoints_queue.append(elem)
            else:
                self._waypoints_queue.append((elem, RoadOption.LANEFOLLOW))
        self._global_plan = True
    
    def run_step(self, debug=True):
        if not self._global_plan and len(self._waypoints_queue) < self.MAX_WP_BUFFER:
            self._compute_next_waypoints(k=100)
        
        while len(self._waypoint_buffer) < self.MAX_WP_BUFFER and self._waypoints_queue:
            self._waypoint_buffer.append(self._waypoints_queue.popleft())

        if len(self._waypoint_buffer) == 0:
            return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        
        transform = self._vehicle.get_transform()

        min_dist = float('inf')
        closest_idx = 0
        search_range = min(15, len(self._waypoint_buffer))
        for i in range(search_range):
            dist = distance_vehicle(self._waypoint_buffer[i][0], transform)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        for _ in range(closest_idx):
            self._waypoint_buffer.popleft()

        if len(self._waypoint_buffer) < 5:
            return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        
        vel = self._vehicle.get_velocity()
        ang_vel = self._vehicle.get_angular_velocity()

        yaw_rad = math.radians(transform.rotation.yaw)
        vx = vel.x * math.cos(yaw_rad) + vel.y * math.sin(yaw_rad)
        vy = -vel.x * math.sin(yaw_rad) + vel.y * math.cos(yaw_rad)
        r_rad = math.radians(ang_vel.z)

        self._ego_state[0] = transform.location.x
        self._ego_state[1] = transform.location.y
        self._ego_state[2] = yaw_rad
        self._ego_state[3] = vx
        self._ego_state[4] = vy
        self._ego_state[5] = r_rad

        num_wp = len(self._waypoint_buffer)
        for i in range(num_wp):
            wp = self._waypoint_buffer[i][0]
            self._wp_x[i] = wp.transform.location.x
            self._wp_y[i] = wp.transform.location.y

        # NMPC Solve
        status, (steer_rad, accel) = self._nmpc_solver.solve(num_wp)

        # C++ 엔진의 판단을 100% 신뢰하되, 특이점 방지용 절대 한계선만 유지
        steer_Rad = np.clip(steer_rad, -0.5, 0.5)
        
        print(f"[{status}] Steer: {steer_rad:+.3f} rad | Accel: {accel:+.3f} m/s^2 | Speed: {vx:+.1f} m/s")
        
        MAX_STEER_PHYSICS = 0.5
        steer_norm = steer_rad / MAX_STEER_PHYSICS

        control = carla.VehicleControl()
        control.steer = float(np.clip(steer_norm, -1.0, 1.0))
        control.gear = 1
        control.manual_gear_shift = False
        control.hand_brake = False

        if accel >= 0.0:
            control.throttle = float(np.clip(accel / 5.0, 0.0, 1.0))
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = float(np.clip(abs(accel) / 10.0, 0.0, 1.0))

        if debug and len(self._waypoint_buffer) > 0:
            draw_waypoints(self._vehicle.get_world(), [self._waypoint_buffer[0][0]], transform.location.z + 1.0)
        return control