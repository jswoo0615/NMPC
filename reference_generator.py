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

        # 2. 다단계 누적 곡률 탐지 (Multi-horizon Accumulated Curvature)
        # 인접 웨이포인트 쌍의 각도 변화를 개별 측정하고 누적하여
        # 교차로의 급격한 방향 전환과 완만한 긴 곡선 모두를 감지합니다.
        max_segment_angle = 0.0
        accumulated_angle = 0.0
        scan_range = min(50, num_wp - 1)
        for i in range(scan_range):
            v1 = waypoint_buffer[i][0].transform.get_forward_vector()
            v2 = waypoint_buffer[i + 1][0].transform.get_forward_vector()
            vec1 = np.array([v1.x, v1.y])
            vec2 = np.array([v2.x, v2.y])
            n1 = np.linalg.norm(vec1)
            n2 = np.linalg.norm(vec2)
            if n1 > 1e-6 and n2 > 1e-6:
                dot = np.clip(np.dot(vec1 / n1, vec2 / n2), -1.0, 1.0)
                seg_angle = math.degrees(math.acos(dot))
                max_segment_angle = max(max_segment_angle, seg_angle)
                accumulated_angle += seg_angle
        # 단일 급회전(교차로)과 완만한 누적 곡선 모두 감지
        angle_diff = max(max_segment_angle, accumulated_angle * 0.3)
            
        # 연속 함수로 속도 감속 — 불연속 경계(if/else) 제거
        # angle_diff 5° 이하: 감속 없음, 35° 이상: 최대 감속, 그 사이 선형 보간
        speed_reduction_factor = min(1.0, max(0.0, (angle_diff - 5.0) / 30.0))
        min_speed_ms = 20.0 / 3.6  # 최저 속도 20 km/h
        raw_target_v = self._base_target_speed - speed_reduction_factor * (self._base_target_speed - min_speed_ms)
        raw_target_v = max(min_speed_ms, raw_target_v)
        dynamic_R_SteerRate = 1200.0 - speed_reduction_factor * 700.0
        dynamic_R_SteerRate = max(300.0, dynamic_R_SteerRate)
            
        if self._smoothed_target_v is None:
            self._smoothed_target_v = max(5.0, vx) 
            
        # 속도 댐핑: 기존 alpha_v=0.02는 90% 도달에 ~6초 소요 → 0.2로 올려 ~0.5초로 단축
        alpha_v = 0.2
        self._smoothed_target_v = (alpha_v * raw_target_v) + ((1.0 - alpha_v) * self._smoothed_target_v)

        # 3. Adaptive KKT Weights (curvature‑aware + lane-change-aware)
        curvature_factor = 1.0
        steer_penalty = 10.0 
        # Q_kappa_track: 곡률이 큰 구간에서는 곡률 추종 페널티를 줄여
        # Q_D(횡오차 보정)와의 충돌을 방지합니다.
        q_kappa_track = max(200.0, 1500.0 - angle_diff * 30.0)

        # [차선 변경 감지] 직선 구간에서 d_error가 큰 경우 = 차선 변경 중
        # Q_kappa_track이 조향을 0으로 강제 고정하는 것을 완화하여
        # 솔버가 횡방향 보정을 위한 조향을 생성할 수 있도록 합니다.
        # d_error 0.3m부터 감소 시작, 1.5m 이상이면 최대 80% 감소
        d_error_lane_change = min(1.0, max(0.0, (d_error - 0.3) / 1.2))
        q_kappa_track = q_kappa_track * (1.0 - 0.8 * d_error_lane_change)

        if angle_diff > 20.0:
            curvature_factor = 2.0
            steer_penalty = 150.0
            dynamic_R_SteerRate = max(300.0, dynamic_R_SteerRate * 0.5)

        # Q_D 및 R_Accel/R_AccelRate: 연속 함수로 전환 (if/else 불연속 제거)
        # d_error가 커질수록 횡오차 보정 가중치를 점진적으로 올립니다.
        # 이전의 d_error > 0.2 이진 분기는 "머뭇거림 → 급전환" 패턴의 원인이었습니다.
        d_blend = min(1.0, max(0.0, (d_error - 0.1) / 0.4))  # 0.1m~0.5m 사이 선형 보간
        q_d_val = 150.0 + d_blend * 50.0          # 150 → 200
        r_accel_val = 50.0 + d_blend * 50.0       # 50 → 100
        r_accel_rate_val = 1000.0 + d_blend * 1000.0  # 1000 → 2000

        # 차선 변경 시 R_SteerRate도 완화: 솔버가 조향 변화를 허용하도록
        if d_error_lane_change > 0.1:
            dynamic_R_SteerRate = dynamic_R_SteerRate * (1.0 - 0.5 * d_error_lane_change)
            dynamic_R_SteerRate = max(200.0, dynamic_R_SteerRate)

        adaptive_opt = {
            'dt': 0.05,
            'Q_D': q_d_val,
            'R_Steer': steer_penalty,
            'R_SteerRate': float(dynamic_R_SteerRate),
            'R_Accel': r_accel_val,
            'R_AccelRate': r_accel_rate_val,
            'Q_kappa_track': q_kappa_track,
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