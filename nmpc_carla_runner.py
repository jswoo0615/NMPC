#!/usr/bin/env python

import sys
import os
import random
import time
import math
import numpy as np
import carla

# CARLA AGENT 경로 추가
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '...'))
    from agent.navigation.global_route_planner import GlobalRoutePlanner
    from nmpc_core_planner import LocalPlanner

except ImportError as e:
    print(f"Error importing modules : {e}")
    sys.exit(1)

def main():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 강제 동기화 모드
    # NMPC C++ 엔진의 dt(0.05)와 CARLA의 물리 엔진 dt를 일치시켜야 적분기 발산 방지
    settings = world.get_settings()
    original_settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.01
    world.apply_settings(settings)

    ego_vehicle = None

    try:
        # 1. Ego Vehicle Spawn (Tesla Model 3 또는 유사 동역학 차량)
        # 지면 충돌 방지를 위해 Z축 0.5m 상향 조정
        blueprint_library = world.get_blueprint_library()
        ego_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        ego_bp.set_attribute('role_name', 'ego')

        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        ego_vehicle = world.spawn_actor(ego_bp, spawn_point)

        # 물리 엔진 안정화 (중력 낙하 및 서스펜션 착지 대기)
        print("Waking up Physics Engine and setting suspension")
        for _ in range(20):
            world.tick()

        # 초기 변속기 강제 체결 및 주차 브레이크 해제
        wake_up_control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=False, manula_gear_shift=False)
        ego_vehicle.apply_control(wake_up_control)
        world.tick()

        # 2. Global Route Planner 초기화 및 경로 생성
        # 거시적인 맵 해상도를 2.0m 단위로 설정
        grp = GlobalRoutePlanner(world.get_map(), 2.0)

        destination = random.choice(spawn_points).location
        route = grp.trace_route(ego_vehicle.get_location(), destination)

        # 튜플 (waypoint, RoadOption) 중 waypoint만 추출하여 NMPC에 전달
        global_plan = [wp for wp, _ in route]

        print(f"Global Route Generated: {len(global_plan)} waypoints.")

        # 3. NMPC Local Planner 초기화 (Zero-copy Pybind11 연결부)
        opt_dict = {
            'target_speed': 50.0,   # km/h
            'dt': 0.01
        }

        nmpc_planner = LocalPlanner(ego_vehicle, opt_dict=opt_dict)
        nmpc_planner.set_global_plan(global_plan)

        print("NMPC Engine Initialized. Engaging Closed-loop Control")

        # 4. Main Control Loop
        spectator = world.get_spectator()

        # 메인 루프 진입 전 초기화
        cam_transform = spectator.get_transform()

        while True:
            world.tick()

            transform = ego_vehicle.get_transform()
            target_location = transform.location + carla.Location(z=3.0) - transform.get_forward_vector() * 5.0
            target_rotation = transform.rotation
            cam_rotation = cam_transform.rotation

            # 최단 경로 각도 보간 (Wrap-around 붕괴 방지)
            diff_yaw = (target_rotation.yaw - cam_rotation.yaw)
            diff_yaw = (diff_yaw + 180.0) % 360.0 - 180.0       # -180 ~ 180 사이의 최단 각도 추출
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

            # 목적지 도달 검사 (Waypoints Queue와 Buffer가 모두 소진 시 종료)
            if len(nmpc_planner._waypoint_queue) == 0 and len(nmpc_planner._waypoint_buffer) == 0:
                print("Destination Reached. NMPC Execution Terminated")
                break

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nSimulation failed: {e}")
    finally:
        # 시뮬레이터 설정 원복 및 액터 파괴
        print("Cleaning up resources...")
        settings = original_settings
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        
        if ego_vehicle is not None:
            ego_vehicle.destroy()

if __name__ == '__main__':
    main()