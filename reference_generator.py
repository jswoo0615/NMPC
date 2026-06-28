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
            
        # 곡선 구간 속도 프로파일 및 조향 레이트 설정
        if angle_diff > 15.0:
            # 곡선 감속: 기존 (angle_diff * 0.4)은 너무 공격적이어서 2.5 m/s까지 떨어졌음
            # 완화하여 실용적인 곡선 속도를 유지합니다 (하한 8.0 m/s ≈ 29 km/h)
            raw_target_v = max(8.0, self._base_target_speed - (angle_diff * 0.15))
            dynamic_R_SteerRate = max(500.0, 1500.0 - (angle_diff * 20.0))
        else:
            raw_target_v = self._base_target_speed
            dynamic_R_SteerRate = 800.0
            
        if self._smoothed_target_v is None:
            self._smoothed_target_v = max(5.0, vx) 
            
        # 속도 댐핑: 기존 alpha_v=0.02는 90% 도달에 ~6초 소요 → 0.2로 올려 ~0.5초로 단축
        alpha_v = 0.2
        self._smoothed_target_v = (alpha_v * raw_target_v) + ((1.0 - alpha_v) * self._smoothed_target_v)

        # 3. Adaptive KKT Weights (curvature‑aware)
        curvature_factor = 1.0
        steer_penalty = 10.0 
        if angle_diff > 20.0:
            curvature_factor = 2.0
            steer_penalty = 150.0
            dynamic_R_SteerRate = max(300.0, dynamic_R_SteerRate * 0.5)

        if d_error > 0.2:
            adaptive_opt = {
                'dt': 0.05,
                'Q_D': 200.0,
                'R_Steer': steer_penalty,
                'R_SteerRate': float(dynamic_R_SteerRate),
                'R_Accel': 100.0,
                'R_AccelRate': 2000.0,
                'target_speed': self._smoothed_target_v
            }
        else:
            adaptive_opt = {
                'dt': 0.05,
                # Q_D 80→150: 차선 변경 시에도 경로를 놓치지 않을 정도의 추종력 확보
                # (잔진동은 solve decimation이 처리하므로 Q_D를 올려도 안전)
                'Q_D': 150.0,
                'R_Steer': steer_penalty,
                'R_SteerRate': float(dynamic_R_SteerRate),
                'R_Accel': 50.0,
                'R_AccelRate': 1000.0,
                'target_speed': self._smoothed_target_v
            }

        # 4. 공간 다중 패스 필터 (Multi-pass Laplacian Smoothing)
        raw_x = [waypoint_buffer[i][0].transform.location.x for i in range(num_wp)]
        raw_y = [waypoint_buffer[i][0].transform.location.y for i in range(num_wp)]

        smooth_x = list(raw_x)
        smooth_y = list(raw_y)

        if vx > 1.0:
            # 경계점 스무딩 (웨이포인트 소비 시 궤적 불연속 방지)
            # 패스 횟수는 3으로 제한 — 5패스 이상은 차선 변경 등 큰 횡이동 시 궤적을 과도하게 왜곡합니다.
            for _ in range(3):
                temp_x = list(smooth_x)
                temp_y = list(smooth_y)

                # 경계점 (index 1) - 3점 윈도우
                if num_wp > 3:
                    temp_x[1] = (smooth_x[0] + smooth_x[1] + smooth_x[2]) / 3.0
                    temp_y[1] = (smooth_y[0] + smooth_y[1] + smooth_y[2]) / 3.0

                # 내부 점 - 5점 윈도우
                for i in range(2, num_wp - 2):
                    temp_x[i] = sum(smooth_x[i-2:i+3]) / 5.0
                    temp_y[i] = sum(smooth_y[i-2:i+3]) / 5.0

                # 경계점 (index num_wp-2) - 3점 윈도우
                if num_wp > 3:
                    temp_x[num_wp-2] = (smooth_x[num_wp-3] + smooth_x[num_wp-2] + smooth_x[num_wp-1]) / 3.0
                    temp_y[num_wp-2] = (smooth_y[num_wp-3] + smooth_y[num_wp-2] + smooth_y[num_wp-1]) / 3.0

                smooth_x = temp_x
                smooth_y = temp_y

            # [안전 장치] 스무딩으로 인한 과도한 궤적 왜곡 방지 (최대 1.0m 이내)
            # 차선 변경(1→4차선 등) 시 Laplacian 스무딩이 전환 구간을 과도하게 깎아
            # 궤적이 가드레일 쪽으로 밀릴 수 있습니다.
            MAX_SMOOTH_DEVIATION = 1.0
            for i in range(num_wp):
                dx = smooth_x[i] - raw_x[i]
                dy = smooth_y[i] - raw_y[i]
                dev = math.sqrt(dx * dx + dy * dy)
                if dev > MAX_SMOOTH_DEVIATION:
                    scale = MAX_SMOOTH_DEVIATION / dev
                    smooth_x[i] = raw_x[i] + dx * scale
                    smooth_y[i] = raw_y[i] + dy * scale

        return num_wp, smooth_x, smooth_y, adaptive_opt