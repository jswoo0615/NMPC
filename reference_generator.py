#!/usr/bin/env python
import math
import numpy as np

class ReferenceGenerator:
    def __init__(self, base_target_speed=5.0):
        self._base_target_speed = base_target_speed
        self._smoothed_target_v = None

    def set_target_speed(self, speed):
        self._base_target_speed = speed

    def generate_reference(self, ego_state, waypoint_buffer, vx):
        """
        CARLA Waypoint Buffer를 기반으로 NMPC가 추종할 부드러운 스플라인 궤적과
        상황 인지형 가중치 (Adaptive Weights)를 도출합니다
        """
        num_wp = len(waypoint_buffer)
        if num_wp < 5:
            return num_wp, [], [], {}
        
        # 1. 횡방향 오차 (d) 도출
        wp_transform = waypoint_buffer[0][0].transform
        wp_loc = wp_transform.location
        wp_yaw_rad = math.radians(wp_transform.rotation.yaw)

        dx = ego_state[0] - wp_loc.x
        dy = ego_state[1] - wp_loc.y
        d_error = abs(-dx * math.sin(wp_yaw_rad) + dy * math.cos(wp_yaw_rad))

        # 2. 미래 곡률 기반 속도 프로파일링 (Lookahead Speed Preview)
        angle_diff = 0.0
        lookahead_idx = min(30, num_wp - 1)
        if lookahead_idx > 5:
            wp_now = waypoint_buffer[0][0].transform.get_forward_vector()
            wp_future = waypoint_buffer[lookahead_idx][0].transform.get_forward_vector()

            vec1 = np.array([wp_now.x, wp_now.y])
            vec2 = np.array([wp_future.x, wp_future.y])
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            vec1 = vec1 / norm1 if norm1 > 1e-6 else vec1
            vec2 = vec2 / norm2 if norm2 > 1e-6 else vec2

            dot_prod = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
            angle_diff = math.degrees(math.acos(dot_prod))

        if angle_diff > 15.0:
            raw_target_v = max(6.0, self._base_target_speed - (angle_diff * 0.4))
            dynamic_R_SteerRate = max(2000.0, 5000.0 - (angle_diff * 40.0))
        else:
            raw_target_v = self._base_target_speed
            dynamic_R_SteerRate = 5000.0

        if self._smoothed_target_v is None:
            self._smoothed_target_v = max(5.0, vx)

        alpha_v = 0.02
        self._smoothed_target_v = (alpha_v * raw_target_v) + ((1.0 - alpha_v) * self._smoothed_target_v)

        # 3. 상황 인지형 동적 가중치 (Adaptive KKT Weights) 생성
        if d_error > 0.2:
            adaptive_opt = {
                'dt': 0.05,
                'Q_D': 50.0,
                'R_Steer': 500.0,
                'R_SteerRate': 5000.0,
                'R_Accel': 100.0,
                'R_AccelRate': 2000.0,
                'target_speed': self._smoothed_target_v
            }
        else:
            adaptive_opt = {
                'dt': 0.05,
                'Q_D': 200.0,
                'R_Steer': 50.0,
                'R_SteerRate': float(dynamic_R_SteerRate),
                'R_Accel': 50.0,
                'R_AccelRate': 1000.0,
                'target_speed': self._smoothed_target_v
            }

        # 4. 공간 다중 패스 필터 (Multi-pass Laplacian Smoothing)
        raw_x = [waypoint_buffer[i][0].tramsform.location.x for i in range(num_wp)]
        raw_y = [waypoint_buffer[i][0].transform.location.y for i in range(num_wp)]

        smooth_x = list(raw_x)
        smooth_y = list(raw_y)

        if vx > 1.0:
            for _ in range(3):
                temp_x = list(smooth_x)
                temp_y = list(smooth_y)

                for i in range(2, num_wp - 2):
                    temp_x[i] = sum(smooth_x[i-2:i+3]) / 5.0
                    temp_y[i] = sum(smooth_y[i-2:i+3]) / 5.0
                smooth_x = temp_x
                smooth_y = temp_y

        return num_wp, smooth_x, smooth_y, adaptive_opt