#!/usr/bin/env python
"""
NMPC CARLA Runner — Scenario Load 및 직접 실행 지원

사용법:
    # 기본: 랜덤 경로
    python nmpc_carla_runner.py

    # 시나리오 JSON 로드
    python nmpc_carla_runner.py --scenario scenario_s219_e231.json

    # Spawn Point 직접 지정
    python nmpc_carla_runner.py --start 219 --end 231

    # 맵 지정
    python nmpc_carla_runner.py --map Town04 --start 219 --end 231

    # 목표 속도 변경
    python nmpc_carla_runner.py --speed 30
"""

import sys
import os
import json
import random
import time
import math
import argparse
import numpy as np
import csv
import carla

# CARLA AGENTS 경로 추가
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'carla'))
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from nmpc_core_planner import LocalPlanner

except ImportError as e:
    print(f"Error importing modules : {e}")
    sys.exit(1)


def load_scenario(filepath):
    """JSON 시나리오 파일을 로드하여 start/end spawn index와 메타데이터를 반환"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"시나리오 로드: {filepath}")
    print(f"  맵: {data.get('map', 'N/A')}")
    print(f"  경로: Spawn[{data['start_spawn_idx']}] → Spawn[{data['end_spawn_idx']}]")
    print(f"  웨이포인트: {data['total_waypoints']}개, 거리: {data['total_distance']:.1f}m")
    print(f"  교차로: {data['num_junctions']}개")
    
    if data.get('junctions_summary'):
        for jct in data['junctions_summary']:
            print(f"    Junction {jct['junction_id']}: max_k={jct['max_curvature']:.4f}, dist={jct['distance']:.1f}m")
    
    return data['start_spawn_idx'], data['end_spawn_idx'], data


class RunMetrics:
    """실행 중 성능 지표를 수집하는 클래스"""
    def __init__(self):
        self.tick_count = 0
        self.fallback_count = 0
        self.max_kkt_error = 0.0
        self.max_lateral_accel = 0.0
        self.max_steer_rate = 0.0
        self.collision_detected = False
        self.completed = False
        self.error = 'UNKNOWN'
        self.start_time = time.time()
        self.steer_history = []
        self.speed_history = []
        self.accel_history = []

    def update(self, status, kkt_err, steer, accel, speed):
        self.tick_count += 1
        if 'Fallback' in status or 'FALLBACK' in status:
            self.fallback_count += 1
        self.max_kkt_error = max(self.max_kkt_error, kkt_err)
        self.steer_history.append(steer)
        self.speed_history.append(speed)
        self.accel_history.append(accel)
        
        if len(self.steer_history) >= 2:
            rate = abs(self.steer_history[-1] - self.steer_history[-2])
            self.max_steer_rate = max(self.max_steer_rate, rate)

    def summary(self):
        elapsed = time.time() - self.start_time
        avg_speed = np.mean(self.speed_history) if self.speed_history else 0.0
        max_accel_mag = max(abs(a) for a in self.accel_history) if self.accel_history else 0.0
        
        return {
            'completed': self.completed,
            'error': self.error,
            'elapsed_sec': round(elapsed, 1),
            'sim_ticks': self.tick_count,
            'fallback_count': self.fallback_count,
            'max_kkt_error': round(self.max_kkt_error, 4),
            'max_steer_rate': round(self.max_steer_rate, 4),
            'max_accel': round(max_accel_mag, 2),
            'avg_speed_ms': round(avg_speed, 1),
            'collision': self.collision_detected
        }


def run_scenario(client, world, start_idx=None, end_idx=None, 
                 target_speed=50.0, max_ticks=30000, collect_metrics=False, log_dir=None):
    """
    단일 시나리오를 실행합니다.
    """
    spawn_points = world.get_map().get_spawn_points()
    
    if start_idx is not None:
        if start_idx < 0 or start_idx >= len(spawn_points):
            raise ValueError(f"start_idx {start_idx}가 범위 밖 (0~{len(spawn_points)-1})")
        spawn_point = spawn_points[start_idx]
    else:
        spawn_point = random.choice(spawn_points)
        start_idx = spawn_points.index(spawn_point)
    
    if end_idx is not None:
        if end_idx < 0 or end_idx >= len(spawn_points):
            raise ValueError(f"end_idx {end_idx}가 범위 밖 (0~{len(spawn_points)-1})")
        destination = spawn_points[end_idx].location
    else:
        destination = random.choice(spawn_points).location
    
    settings = world.get_settings()
    original_settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.01
    world.apply_settings(settings)
    
    ego_vehicle = None
    metrics = RunMetrics() if collect_metrics else None
    csv_file = None
    csv_writer = None
    
    try:
        if log_dir is not None:
            csv_file = open(os.path.join(log_dir, 'solver_log.csv'), 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['tick', 'x', 'y', 'speed_ms', 'steer', 'accel', 'kkt_error', 'min_slack', 'max_lam', 'solver_status'])

        blueprint_library = world.get_blueprint_library()
        ego_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        ego_bp.set_attribute('role_name', 'ego')
        
        ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
        
        collision_sensor = None
        if metrics:
            collision_bp = blueprint_library.find('sensor.other.collision')
            collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)
            def on_collision(event):
                metrics.collision_detected = True
            collision_sensor.listen(on_collision)
        
        print("Waking up Physics Engine and setting suspension")
        for _ in range(20):
            world.tick()
        
        wake_up_control = carla.VehicleControl(
            throttle=0.0, brake=1.0, steer=0.0, 
            hand_brake=False, manual_gear_shift=False
        )
        ego_vehicle.apply_control(wake_up_control)
        world.tick()
        
        grp = GlobalRoutePlanner(world.get_map(), 2.0)
        route = grp.trace_route(ego_vehicle.get_location(), destination)
        global_plan = [wp for wp, _ in route]
        
        total_dist = 0.0
        for i in range(1, len(global_plan)):
            l0 = global_plan[i-1].transform.location
            l1 = global_plan[i].transform.location
            total_dist += math.hypot(l1.x - l0.x, l1.y - l0.y)
            
        needed_ticks = int((total_dist / 8.0) * 100) + 6000
        max_ticks = max(max_ticks, needed_ticks)
        
        print(f"Route: Spawn[{start_idx}] → Spawn[{end_idx}], {len(global_plan)} waypoints ({total_dist:.1f}m)")
        print(f"Dynamic Timeout set to {max_ticks} ticks")
        
        opt_dict = {
            'target_speed': target_speed,
            'dt': 0.01
        }
        
        nmpc_planner = LocalPlanner(ego_vehicle, opt_dict=opt_dict)
        nmpc_planner.set_global_plan(global_plan)
        
        print("NMPC Engine Initialized. Engaging Closed-loop Control")
        
        spectator = world.get_spectator()
        cam_transform = spectator.get_transform()
        tick = 0
        consecutive_fallbacks = 0
        
        while tick < max_ticks:
            world.tick()
            tick += 1
            
            transform = ego_vehicle.get_transform()
            target_location = transform.location + carla.Location(z=3.0) - transform.get_forward_vector() * 5.0
            target_rotation = transform.rotation
            cam_rotation = cam_transform.rotation
            
            diff_yaw = (target_rotation.yaw - cam_rotation.yaw)
            diff_yaw = (diff_yaw + 180.0) % 360.0 - 180.0       
            new_yaw = cam_rotation.yaw + diff_yaw * 0.1
            
            diff_pitch = (target_rotation.pitch - cam_rotation.pitch)
            diff_pitch = (diff_pitch + 180.0) % 360.0 - 180.0
            new_pitch = cam_rotation.pitch + diff_pitch * 0.1
            
            diff_roll = (target_rotation.roll - cam_rotation.roll)
            diff_roll = (diff_roll + 180.0) % 360.0 - 180.0
            new_roll = cam_rotation.roll + diff_roll * 0.1
            
            cam_transform = carla.Transform(
                target_location,
                carla.Rotation(pitch=new_pitch, yaw=new_yaw, roll=new_roll)
            )
            spectator.set_transform(cam_transform)
            
            control = nmpc_planner.run_step(debug=True)
            ego_vehicle.apply_control(control)
            
            kkt_err = 0.0
            min_slack = 0.0
            max_lam = 0.0
            status_msg = "UNKNOWN"
            
            if hasattr(nmpc_planner, 'latest_metrics'):
                m_data = nmpc_planner.latest_metrics
                kkt_err = m_data.get('kkt_error', 0.0)
                min_slack = m_data.get('min_slack', 0.0)
                max_lam = m_data.get('max_lambda', 0.0)
                status_msg = m_data.get('status', 'ERROR')
            else:
                status_msg = getattr(nmpc_planner, 'last_status', 'OK')
                kkt_err = getattr(nmpc_planner, 'last_kkt_err', 0.0)
                
            steer = control.steer
            accel_val = control.throttle - control.brake
            current_vel = ego_vehicle.get_velocity()
            speed = math.hypot(current_vel.x, current_vel.y)
            
            if metrics:
                metrics.update(status_msg, kkt_err, steer, accel_val, speed)
            
            if csv_writer:
                csv_writer.writerow([
                    tick, transform.location.x, transform.location.y, speed,
                    steer, accel_val, kkt_err, min_slack, max_lam, status_msg
                ])

            if metrics and metrics.collision_detected:
                metrics.completed = False
                metrics.error = 'COLLISION'
                break
                
            if status_msg == 'FALLBACK':
                consecutive_fallbacks += 1
                if consecutive_fallbacks > 10:
                    if metrics:
                        metrics.completed = False
                        metrics.error = 'SOLVER_FAIL'
                    break
            else:
                consecutive_fallbacks = 0
                
            # [아키텍트의 수술: 유클리디안 거리 제거, 크로스 트랙 에러 및 유예 기간 적용]
            if tick > 50 and len(nmpc_planner._waypoint_buffer) > 0:
                wp_transform = nmpc_planner._waypoint_buffer[0][0].transform
                wp_loc = wp_transform.location
                wp_yaw_rad = math.radians(wp_transform.rotation.yaw)
                
                dx = transform.location.x - wp_loc.x
                dy = transform.location.y - wp_loc.y
                
                cross_track_error = abs(-dx * math.sin(wp_yaw_rad) + dy * math.cos(wp_yaw_rad))
                
                if cross_track_error > 4.0:
                    if metrics:
                        metrics.completed = False
                        metrics.error = 'OFF_ROUTE'
                    break
            
            if len(nmpc_planner._waypoint_queue) == 0 and len(nmpc_planner._waypoint_buffer) <= 5:
                print("Destination Reached. NMPC Execution Terminated")
                if metrics:
                    metrics.completed = True
                    metrics.error = 'NONE'
                break
        
        if tick >= max_ticks:
            print(f"Timeout: {max_ticks} ticks에 도달. 강제 종료.")
            if metrics:
                metrics.completed = False
                metrics.error = 'TIMEOUT'
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nSimulation failed: {e}")
        import traceback
        traceback.print_exc()
        if metrics:
            metrics.completed = False
            metrics.error = f'EXCEPTION: {str(e)}'
    finally:
        print("Cleaning up resources...")
        world.apply_settings(original_settings)
        
        if csv_file:
            csv_file.close()
            
        if collision_sensor is not None:
            collision_sensor.destroy()
        if ego_vehicle is not None:
            ego_vehicle.destroy()
        
        for _ in range(10):
            try:
                world.tick()
            except:
                pass
    
    return metrics.summary() if metrics else None


def main():
    parser = argparse.ArgumentParser(description='NMPC CARLA Runner')
    parser.add_argument('--scenario', type=str, default=None,
                        help='JSON 시나리오 파일 경로 (carla_route_inspector에서 export)')
    parser.add_argument('--start', type=int, default=None,
                        help='출발 Spawn Point 인덱스')
    parser.add_argument('--end', type=int, default=None,
                        help='도착 Spawn Point 인덱스')
    parser.add_argument('--map', type=str, default=None,
                        help='CARLA 맵 이름 (예: Town04)')
    parser.add_argument('--speed', type=float, default=60.0,
                        help='목표 속도 (km/h, 기본 60)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='CARLA 서버 호스트')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA 서버 포트')
    args = parser.parse_args()
    
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    
    if args.map:
        print(f"맵 로딩: {args.map}")
        client.load_world(args.map)
        time.sleep(2.0)
    
    world = client.get_world()
    
    start_idx = args.start
    end_idx = args.end
    
    if args.scenario:
        start_idx, end_idx, _ = load_scenario(args.scenario)
    
    run_scenario(
        client, world,
        start_idx=start_idx, 
        end_idx=end_idx,
        target_speed=args.speed
    )


if __name__ == '__main__':
    main()