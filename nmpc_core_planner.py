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
        
        # [아키텍트의 수술: Speed & Steering Profiling 및 스무딩 통합본]
        # 변수 초기화: 웨이포인트 버퍼가 고갈되어도 로직이 붕괴하지 않도록 기본값 할당
        angle_diff = 0.0 
        
        lookahead_idx = min(15, len(self._waypoint_buffer) - 1)
        if lookahead_idx > 5:
            wp_now = self._waypoint_buffer[0][0].transform.get_forward_vector()
            wp_future = self._waypoint_buffer[lookahead_idx][0].transform.get_forward_vector()
            
            dot_prod = np.clip(wp_now.x * wp_future.x + wp_now.y * wp_future.y, -1.0, 1.0)
            angle_diff = math.degrees(math.acos(dot_prod))
            
        # 곡률에 따른 타겟 속도 및 조향 관성 계산
        if angle_diff > 10.0:
            raw_target_v = max(4.0, 25.0 - (angle_diff * 0.5))
            dynamic_R_SteerRate = max(100.0, 1500.0 - (angle_diff * 50.0))
        else:
            raw_target_v = 25.0
            dynamic_R_SteerRate = 1500.0
            
        # 초기화가 안 되어 있다면 현재 속도로 초기화 하되, 
        # 정지 상태 출발 시 최소한의 초기 타겟 속도(5.0m/s)를 보장하여 솔버가 가속 의지를 갖게 함
        if not hasattr(self, '_smoothed_target_v'):
            self._smoothed_target_v = max(5.0, vx) 
            
        # EMA 필터
        alpha_v = 0.02 
        self._smoothed_target_v = (alpha_v * raw_target_v) + ((1.0 - alpha_v) * self._smoothed_target_v)
            
        self._nmpc_solver.set_target_speed(self._smoothed_target_v)
        
        adaptive_opt = {
            'R_SteerRate': float(dynamic_R_SteerRate)
        }
        self._nmpc_solver.update_config(adaptive_opt)

        # (NMPC Solve 호출 직전의 Waypoint 좌표 할당 부분)
        num_wp = len(self._waypoint_buffer)
        
        # 1. CARLA의 날것(Raw) 좌표를 임시 리스트에 추출
        raw_x = [self._waypoint_buffer[i][0].transform.location.x for i in range(num_wp)]
        raw_y = [self._waypoint_buffer[i][0].transform.location.y for i in range(num_wp)]

        # [아키텍트의 수술: Reference Generator - 공간 로우패스 필터]
        # 교차로의 90도 꺾임(Kink) 현상을 방지하기 위해 윈도우 사이즈 5의 이동 평균 필터를 통과시킵니다.
        # 직각 모서리가 자동차가 실제로 돌 수 있는 부드러운 곡선(Curve)으로 물리적으로 다듬어집니다.
        # [아키텍트의 수술: 적응형 공간 필터]
        # 저속(출발/정차)에서는 필터를 끄고 원본 웨이포인트를 사용하여 응답성을 살립니다.
        # 고속(곡률 변화가 크고 반응이 늦은 구간)에서만 필터를 작동시킵니다.
        
        window = 5
        half_w = window // 2
        
        # 필터링 여부 결정: 속도가 2m/s(7.2km/h) 미만이면 필터링 없이 원본 사용
        if vx < 2.0:
            for i in range(num_wp):
                self._wp_x[i] = raw_x[i]
                self._wp_y[i] = raw_y[i]
        else:
            for i in range(num_wp):
                if i < half_w or i >= num_wp - half_w:
                    self._wp_x[i] = raw_x[i]
                    self._wp_y[i] = raw_y[i]
                else:
                    self._wp_x[i] = sum(raw_x[i-half_w : i+half_w+1]) / float(window)
                    self._wp_y[i] = sum(raw_y[i-half_w : i+half_w+1]) / float(window)

        # NMPC Solve 
        status, (steer_rad, accel) = self._nmpc_solver.solve(num_wp)

        # C++ 엔진의 판단을 100% 신뢰하되, 특이점 방지용 절대 한계선만 유지
        steer_Rad = np.clip(steer_rad, -0.5, 0.5)
        
        print(f"[{status}] Steer: {steer_rad:+.3f} rad | Accel: {accel:+.3f} m/s^2 | Speed: {vx:+.1f} m/s")
        
        MAX_STEER_PHYSICS = 0.5
        steer_norm = steer_rad / MAX_STEER_PHYSICS

        control = carla.VehicleControl()
        control.steer = float(np.clip(steer_norm, -1.0, 1.0))
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