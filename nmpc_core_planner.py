#!/usr/bin/env python

import math
from enum import Enum
from collections import deque
import random
import numpy as np

import carla
from agents.tools.misc import distance_vehicle, draw_waypoints
import nmpc_core

from reference_generator import ReferenceGenerator

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
        
        self._target_speed = 5.0        
        self._sampling_radius = 1.0

        self._waypoint_queue = deque(maxlen=20000)
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

        self._ref_gen = ReferenceGenerator(base_target_speed=self._target_speed)
        self._nmpc_solver.set_target_speed(self._target_speed)

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._waypoint_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))
        self._compute_next_waypoints(k=200)

    def set_speed(self, speed):
        self._target_speed = speed / 3.6
        self._ref_gen.set_target_speed(self._target_speed)

    def _compute_next_waypoints(self, k=1):
        available = self._waypoint_queue.maxlen - len(self._waypoint_queue)
        for _ in range(min(available, k)):
            last_wp = self._waypoint_queue[-1][0]
            next_wps = list(last_wp.next(self._sampling_radius))
            next_wp = next_wps[0] if len(next_wps) == 1 else random.choice(next_wps)
            self._waypoint_queue.append((next_wp, RoadOption.LANEFOLLOW))
    
    def set_global_plan(self, current_plan):
        self._waypoint_queue.clear()
        for elem in current_plan:
            if isinstance(elem, tuple):
                self._waypoint_queue.append(elem)
            else:
                self._waypoint_queue.append((elem, RoadOption.LANEFOLLOW))
        self._global_plan = True
    
    def run_step(self, debug=True):
        if not hasattr(self, '_solve_counter'):
            self._solve_counter = -1
            self._cached_control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=0.0)

        self._solve_counter += 1

        if not self._global_plan and len(self._waypoint_queue) < self.MAX_WP_BUFFER:
            self._compute_next_waypoints(k=100)
        
        while len(self._waypoint_buffer) < self.MAX_WP_BUFFER and self._waypoint_queue:
            self._waypoint_buffer.append(self._waypoint_queue.popleft())

        if len(self._waypoint_buffer) == 0:
            return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)

        SOLVE_EVERY_N = 5
        if self._solve_counter % SOLVE_EVERY_N != 0:
            return self._cached_control

        transform = self._vehicle.get_transform()
        ego_loc = transform.location
        ego_forward = transform.get_forward_vector()
        
        # [아키텍트의 수술: 내적 기반 강건한 웨이포인트 통과 판별]
        pop_count = 0
        search_range = min(15, len(self._waypoint_buffer))
        for i in range(search_range):
            wp_loc = self._waypoint_buffer[i][0].transform.location
            vec_to_wp = carla.Vector3D(wp_loc.x - ego_loc.x, wp_loc.y - ego_loc.y, 0.0)
            
            dist = math.hypot(vec_to_wp.x, vec_to_wp.y)
            dot_product = vec_to_wp.x * ego_forward.x + vec_to_wp.y * ego_forward.y
            
            # 물리적으로 거리가 가깝거나, 내적이 음수(차량 뒤로 넘어감)일 경우 통과 판정
            if dist < 1.5 or dot_product < 0.0:
                pop_count = i + 1
            else:
                break
                
        # Fallback: 궤적을 크게 이탈하여 내적 조건에 걸리지 않는 경우 기존 최단거리 백업 적용
        if pop_count == 0 and search_range > 0:
            min_dist = float('inf')
            closest_idx = 0
            for i in range(search_range):
                dist = distance_vehicle(self._waypoint_buffer[i][0], transform)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            pop_count = closest_idx

        for _ in range(pop_count):
            self._waypoint_buffer.popleft()

        # Runner의 종료 조건(<= 5)과 정확히 호응하도록 수정
        if len(self._waypoint_buffer) <= 5:
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
        
        num_wp, ref_x, ref_y, adaptive_opt = self._ref_gen.generate_reference(
            self._ego_state, 
            self._waypoint_buffer, 
            vx
        )

        self._nmpc_solver.update_config(adaptive_opt)
        self._nmpc_solver.set_target_speed(adaptive_opt['target_speed'])

        if not hasattr(self, '_warmup_count'):
            self._warmup_count = 0
        self._warmup_count += 1
        if self._warmup_count <= 10:
            clamped_speed = min(adaptive_opt.get('target_speed', 5.0), 5.0)
            self._nmpc_solver.set_target_speed(clamped_speed)
        
        for i in range(num_wp):
            self._wp_x[i] = ref_x[i]
            self._wp_y[i] = ref_y[i]

        monitor_data, (raw_steer, accel) = self._nmpc_solver.solve(num_wp)
        
        status = monitor_data.get('status', 'ERROR')
        kkt_err = monitor_data.get('kkt_error', 0.0)
        sqp_iter = monitor_data.get('sqp_iter', 0)
        min_slack = monitor_data.get('min_slack', 0.0)
        max_lam = monitor_data.get('max_lambda', 0.0)
        
        if not hasattr(self, '_last_steer_rad'):
            self._last_steer_rad = raw_steer
        
        max_delta = 0.05 
        steer_rad = np.clip(raw_steer, self._last_steer_rad - max_delta, self._last_steer_rad + max_delta)
        self._last_steer_rad = steer_rad
        
        print(f"[{status}|Iter:{sqp_iter}|KKT:{kkt_err:.3f}|Slack:{min_slack:.3f}|Lam:{max_lam:.1f}] Steer: {steer_rad:+.3f} rad | Accel: {accel:+.3f} m/s^2 | Speed: {vx:+.1f} m/s")
        
        MAX_STEER_PHYSICS = 1.22 
        steer_norm = steer_rad / MAX_STEER_PHYSICS

        control = carla.VehicleControl()
        control.steer = float(np.clip(steer_norm, -1.0, 1.0))

        if accel >= 0.0:
            control.throttle = float(np.clip(accel / 5.0, 0.0, 1.0))
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = float(np.clip(abs(accel) / 5.0, 0.0, 1.0))
        
        if debug:
            draw_waypoints(self._vehicle.get_world(), [self._waypoint_buffer[0][0]], transform.location.z + 1.0)
        
        self._cached_control = control
        return control