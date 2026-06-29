#!/usr/bin/env python
"""
CARLA Route Inspector — 맵 웨이포인트 & Trajectory 시각화 진단 도구

사용법:
    python carla_route_inspector.py [--map MAP_NAME]

기능:
    1. 맵 전체를 조감도(Bird's-eye View)로 표시
    2. 모든 Spawn Point를 번호와 함께 시각화
    3. 도로 토폴로지(Waypoint)를 시각화
    4. 두 Spawn Point 사이의 GlobalRoutePlanner 경로를 생성 및 시각화
    5. 교차로(Junction) 웨이포인트를 하이라이트
    6. 생성된 Trajectory 상세 정보(거리, 곡률, 도로ID) 출력
    7. Trajectory를 JSON으로 저장하여 nmpc_carla_runner에서 재사용 가능

제어:
    Interactive CLI에서 명령어를 입력하여 사용합니다.
"""

import sys
import os
import json
import math
import time
import argparse

try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'carla'))
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError as e:
    print(f"Import Error: {e}")
    print("CARLA Python API 또는 agents 패키지 경로를 확인하세요.")
    sys.exit(1)


# ─────────────────────────────────────────────
# 색상 상수
# ─────────────────────────────────────────────
COLOR_SPAWN      = carla.Color(0, 255, 0)      # 초록: Spawn Points
COLOR_SPAWN_TEXT = carla.Color(255, 255, 255)   # 흰색: Spawn 번호 텍스트
COLOR_ROAD       = carla.Color(80, 80, 80)      # 회색: 일반 도로 웨이포인트
COLOR_JUNCTION   = carla.Color(255, 165, 0)     # 주황: 교차로 웨이포인트
COLOR_ROUTE      = carla.Color(0, 120, 255)     # 파랑: 경로
COLOR_ROUTE_JCT  = carla.Color(255, 0, 0)       # 빨강: 경로 중 교차로 구간
COLOR_START      = carla.Color(0, 255, 0)       # 초록: 출발점
COLOR_END        = carla.Color(255, 0, 0)       # 빨강: 도착점
COLOR_HEADING    = carla.Color(255, 255, 0)     # 노랑: 헤딩 방향 화살표

DRAW_LIFE = 60.0  # 드로잉 지속 시간 (초)


class RouteInspector:
    def __init__(self, client, map_name=None):
        self.client = client
        
        if map_name:
            print(f"맵 로딩 중: {map_name}")
            self.client.load_world(map_name)
            time.sleep(2.0)
        
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        self.debug = self.world.get_debug()
        self.spectator = self.world.get_spectator()
        
        self.spawn_points = self.carla_map.get_spawn_points()
        self.grp = GlobalRoutePlanner(self.carla_map, 2.0)
        
        # 현재 활성 경로
        self.current_route = None
        self.current_route_info = None
        
        print(f"맵: {self.carla_map.name}")
        print(f"Spawn Points: {len(self.spawn_points)}개")

    # ─────────────────────────────────────────
    # 조감도 카메라 설정
    # ─────────────────────────────────────────
    def set_bird_eye_view(self, center=None, height=200.0):
        """맵 전체를 내려다보는 조감도 설정"""
        if center is None:
            # 모든 Spawn Point의 중심 계산
            if not self.spawn_points:
                center = carla.Location(0, 0, 0)
            else:
                avg_x = sum(sp.location.x for sp in self.spawn_points) / len(self.spawn_points)
                avg_y = sum(sp.location.y for sp in self.spawn_points) / len(self.spawn_points)
                center = carla.Location(avg_x, avg_y, 0)
        
        self.spectator.set_transform(carla.Transform(
            carla.Location(center.x, center.y, height),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        ))
        print(f"조감도 설정: 중심=({center.x:.0f}, {center.y:.0f}), 높이={height:.0f}m")

    def focus_location(self, location, height=50.0):
        """특정 위치로 카메라를 이동"""
        self.spectator.set_transform(carla.Transform(
            carla.Location(location.x, location.y, height),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        ))

    # ─────────────────────────────────────────
    # Spawn Point 시각화
    # ─────────────────────────────────────────
    def draw_spawn_points(self):
        """모든 Spawn Point를 번호와 함께 시각화"""
        print(f"\n{'='*60}")
        print(f" Spawn Points ({len(self.spawn_points)}개)")
        print(f"{'='*60}")
        print(f"{'Idx':>4} | {'X':>9} | {'Y':>9} | {'Z':>6} | {'Yaw':>7} | Road ID")
        print(f"{'-'*4}-+-{'-'*9}-+-{'-'*9}-+-{'-'*6}-+-{'-'*7}-+-{'-'*10}")
        
        for idx, sp in enumerate(self.spawn_points):
            loc = sp.location
            yaw = sp.rotation.yaw
            
            # 맵에서 가장 가까운 웨이포인트의 road_id 조회
            nearest_wp = self.carla_map.get_waypoint(loc)
            road_id = nearest_wp.road_id if nearest_wp else "N/A"
            is_junction = nearest_wp.is_junction if nearest_wp else False
            jct_mark = " [JCT]" if is_junction else ""
            
            print(f"{idx:>4} | {loc.x:>9.1f} | {loc.y:>9.1f} | {loc.z:>6.1f} | {yaw:>7.1f} | {road_id}{jct_mark}")
            
            # 3D 시각화: 점 + 번호
            draw_loc = carla.Location(loc.x, loc.y, loc.z + 0.5)
            self.debug.draw_point(draw_loc, size=0.3, color=COLOR_SPAWN, life_time=DRAW_LIFE)
            self.debug.draw_string(
                carla.Location(loc.x, loc.y, loc.z + 2.0),
                str(idx), 
                draw_shadow=True, 
                color=COLOR_SPAWN_TEXT, 
                life_time=DRAW_LIFE
            )
            
            # 헤딩 방향 화살표
            fwd = sp.get_forward_vector()
            end_loc = carla.Location(loc.x + fwd.x * 3.0, loc.y + fwd.y * 3.0, loc.z + 0.5)
            self.debug.draw_arrow(
                draw_loc, end_loc,
                thickness=0.1, arrow_size=0.2,
                color=COLOR_HEADING, life_time=DRAW_LIFE
            )

    # ─────────────────────────────────────────
    # 도로 토폴로지 시각화
    # ─────────────────────────────────────────
    def draw_road_topology(self, spacing=5.0):
        """도로 전체 웨이포인트를 시각화 (교차로 하이라이트)"""
        waypoints = self.carla_map.generate_waypoints(spacing)
        
        junction_count = 0
        road_count = 0
        
        for wp in waypoints:
            loc = wp.transform.location
            draw_loc = carla.Location(loc.x, loc.y, loc.z + 0.3)
            
            if wp.is_junction:
                self.debug.draw_point(draw_loc, size=0.15, color=COLOR_JUNCTION, life_time=DRAW_LIFE)
                junction_count += 1
            else:
                self.debug.draw_point(draw_loc, size=0.08, color=COLOR_ROAD, life_time=DRAW_LIFE)
                road_count += 1
        
        print(f"\n도로 토폴로지 시각화 완료:")
        print(f"  일반 도로 웨이포인트 (회색): {road_count}개")
        print(f"  교차로 웨이포인트 (주황): {junction_count}개")
        print(f"  간격: {spacing}m")

    # ─────────────────────────────────────────
    # 경로 생성 및 시각화
    # ─────────────────────────────────────────
    def generate_route(self, start_idx, end_idx):
        """두 Spawn Point 사이의 경로를 생성하고 시각화"""
        if start_idx < 0 or start_idx >= len(self.spawn_points):
            print(f"Error: start_idx {start_idx}가 범위(0~{len(self.spawn_points)-1}) 밖입니다.")
            return None
        if end_idx < 0 or end_idx >= len(self.spawn_points):
            print(f"Error: end_idx {end_idx}가 범위(0~{len(self.spawn_points)-1}) 밖입니다.")
            return None
        
        start_loc = self.spawn_points[start_idx].location
        end_loc = self.spawn_points[end_idx].location
        
        print(f"\n경로 생성 중: Spawn[{start_idx}] → Spawn[{end_idx}]")
        print(f"  출발: ({start_loc.x:.1f}, {start_loc.y:.1f})")
        print(f"  도착: ({end_loc.x:.1f}, {end_loc.y:.1f})")
        
        route = self.grp.trace_route(start_loc, end_loc)
        
        if not route:
            print("Error: 경로를 생성할 수 없습니다.")
            return None
        
        self.current_route = route
        self.current_route_info = self._analyze_route(route, start_idx, end_idx)
        self._draw_route(route)
        self._print_route_info(self.current_route_info)
        
        return route

    def _analyze_route(self, route, start_idx, end_idx):
        """경로를 분석하여 상세 정보를 추출"""
        info = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'total_waypoints': len(route),
            'total_distance': 0.0,
            'segments': [],
            'junctions': [],
            'curvature_profile': [],
            'waypoints_data': []
        }
        
        current_segment = {
            'type': 'road',
            'road_id': None,
            'lane_id': None,
            'start_wp_idx': 0,
            'end_wp_idx': 0,
            'distance': 0.0,
            'is_junction': False,
            'junction_id': None,
            'max_curvature': 0.0
        }
        
        prev_loc = None
        
        for i, (wp, road_option) in enumerate(route):
            loc = wp.transform.location
            yaw = wp.transform.rotation.yaw
            
            # 거리 계산
            if prev_loc:
                seg_dist = math.sqrt(
                    (loc.x - prev_loc.x)**2 + (loc.y - prev_loc.y)**2
                )
                info['total_distance'] += seg_dist
            else:
                seg_dist = 0.0
            
            # 곡률 계산 (3점 기반)
            curvature = 0.0
            if i >= 1 and i < len(route) - 1:
                p0 = route[i-1][0].transform.location
                p1 = loc
                p2 = route[i+1][0].transform.location
                curvature = self._compute_curvature_3pt(p0, p1, p2)
            
            info['curvature_profile'].append(curvature)
            
            # 웨이포인트 데이터 저장
            info['waypoints_data'].append({
                'index': i,
                'x': loc.x,
                'y': loc.y,
                'z': loc.z,
                'yaw': yaw,
                'road_id': wp.road_id,
                'lane_id': wp.lane_id,
                'is_junction': wp.is_junction,
                'junction_id': wp.junction_id if wp.is_junction else -1,
                'road_option': str(road_option),
                'curvature': curvature,
                'cumulative_dist': info['total_distance']
            })
            
            # 세그먼트 변경 감지
            new_segment = False
            if wp.is_junction != current_segment['is_junction']:
                new_segment = True
            elif not wp.is_junction and wp.road_id != current_segment['road_id']:
                new_segment = True
            elif wp.is_junction and wp.junction_id != current_segment.get('junction_id'):
                new_segment = True
            
            if new_segment and i > 0:
                current_segment['end_wp_idx'] = i - 1
                info['segments'].append(current_segment)
                
                if current_segment['is_junction']:
                    info['junctions'].append({
                        'junction_id': current_segment['junction_id'],
                        'wp_range': (current_segment['start_wp_idx'], current_segment['end_wp_idx']),
                        'max_curvature': current_segment['max_curvature'],
                        'distance': current_segment['distance']
                    })
                
                current_segment = {
                    'type': 'junction' if wp.is_junction else 'road',
                    'road_id': wp.road_id,
                    'lane_id': wp.lane_id,
                    'start_wp_idx': i,
                    'end_wp_idx': i,
                    'distance': 0.0,
                    'is_junction': wp.is_junction,
                    'junction_id': wp.junction_id if wp.is_junction else None,
                    'max_curvature': 0.0
                }
            else:
                current_segment['road_id'] = wp.road_id
                current_segment['lane_id'] = wp.lane_id
                current_segment['distance'] += seg_dist
                current_segment['max_curvature'] = max(current_segment['max_curvature'], abs(curvature))
                if wp.is_junction:
                    current_segment['junction_id'] = wp.junction_id
            
            prev_loc = loc
        
        # 마지막 세그먼트
        current_segment['end_wp_idx'] = len(route) - 1
        info['segments'].append(current_segment)
        if current_segment['is_junction']:
            info['junctions'].append({
                'junction_id': current_segment['junction_id'],
                'wp_range': (current_segment['start_wp_idx'], current_segment['end_wp_idx']),
                'max_curvature': current_segment['max_curvature'],
                'distance': current_segment['distance']
            })
        
        return info

    def _compute_curvature_3pt(self, p0, p1, p2):
        """3점 기반 곡률 계산 (Menger curvature)"""
        ax, ay = p1.x - p0.x, p1.y - p0.y
        bx, by = p2.x - p1.x, p2.y - p1.y
        
        cross = abs(ax * by - ay * bx)
        a = math.sqrt(ax*ax + ay*ay)
        b = math.sqrt(bx*bx + by*by)
        cx, cy = p2.x - p0.x, p2.y - p0.y
        c = math.sqrt(cx*cx + cy*cy)
        
        denom = a * b * c
        if denom < 1e-6:
            return 0.0
        
        return 2.0 * cross / denom

    def _draw_route(self, route):
        """경로를 3D 시각화"""
        # 출발점 (큰 초록 구)
        start_loc = route[0][0].transform.location
        self.debug.draw_point(
            carla.Location(start_loc.x, start_loc.y, start_loc.z + 1.0),
            size=0.5, color=COLOR_START, life_time=DRAW_LIFE
        )
        self.debug.draw_string(
            carla.Location(start_loc.x, start_loc.y, start_loc.z + 3.0),
            "START", draw_shadow=True, color=COLOR_START, life_time=DRAW_LIFE
        )
        
        # 도착점 (큰 빨간 구)
        end_loc = route[-1][0].transform.location
        self.debug.draw_point(
            carla.Location(end_loc.x, end_loc.y, end_loc.z + 1.0),
            size=0.5, color=COLOR_END, life_time=DRAW_LIFE
        )
        self.debug.draw_string(
            carla.Location(end_loc.x, end_loc.y, end_loc.z + 3.0),
            "END", draw_shadow=True, color=COLOR_END, life_time=DRAW_LIFE
        )
        
        # 경로 라인
        for i in range(len(route) - 1):
            wp_curr = route[i][0]
            wp_next = route[i+1][0]
            
            loc_curr = wp_curr.transform.location
            loc_next = wp_next.transform.location
            
            z_offset = 0.5
            p1 = carla.Location(loc_curr.x, loc_curr.y, loc_curr.z + z_offset)
            p2 = carla.Location(loc_next.x, loc_next.y, loc_next.z + z_offset)
            
            color = COLOR_ROUTE_JCT if wp_curr.is_junction else COLOR_ROUTE
            
            self.debug.draw_line(p1, p2, thickness=0.15, color=color, life_time=DRAW_LIFE)
            
            # 교차로 진입/이탈 지점에 마커
            if i > 0:
                was_junction = route[i-1][0].is_junction
                if wp_curr.is_junction and not was_junction:
                    self.debug.draw_string(
                        carla.Location(loc_curr.x, loc_curr.y, loc_curr.z + 2.0),
                        f"JCT_IN({wp_curr.junction_id})",
                        draw_shadow=True, color=COLOR_ROUTE_JCT, life_time=DRAW_LIFE
                    )
                elif not wp_curr.is_junction and was_junction:
                    self.debug.draw_string(
                        carla.Location(loc_curr.x, loc_curr.y, loc_curr.z + 2.0),
                        "JCT_OUT",
                        draw_shadow=True, color=COLOR_JUNCTION, life_time=DRAW_LIFE
                    )
        
        print(f"경로 시각화 완료 (지속시간: {DRAW_LIFE}초)")

    def _print_route_info(self, info):
        """경로 분석 결과를 출력"""
        print(f"\n{'='*70}")
        print(f" Trajectory 분석 결과")
        print(f"{'='*70}")
        print(f"  Spawn[{info['start_idx']}] → Spawn[{info['end_idx']}]")
        print(f"  총 웨이포인트: {info['total_waypoints']}개")
        print(f"  총 거리: {info['total_distance']:.1f}m")
        print(f"  세그먼트: {len(info['segments'])}개")
        print(f"  교차로 통과: {len(info['junctions'])}회")
        
        if info['curvature_profile']:
            max_k = max(abs(k) for k in info['curvature_profile'])
            avg_k = sum(abs(k) for k in info['curvature_profile']) / len(info['curvature_profile'])
            print(f"  최대 곡률: {max_k:.4f} (1/m)")
            print(f"  평균 곡률: {avg_k:.4f} (1/m)")
        
        # 세그먼트 테이블
        print(f"\n{'─'*70}")
        print(f"{'Seg':>3} | {'Type':^8} | {'Road/Jct ID':>11} | {'WP Range':>12} | {'Dist(m)':>8} | {'MaxK':>7}")
        print(f"{'─'*3}-+-{'─'*8}-+-{'─'*11}-+-{'─'*12}-+-{'─'*8}-+-{'─'*7}")
        
        for i, seg in enumerate(info['segments']):
            seg_type = "JCT" if seg['is_junction'] else "ROAD"
            id_str = str(seg['junction_id']) if seg['is_junction'] else str(seg['road_id'])
            wp_range = f"{seg['start_wp_idx']}-{seg['end_wp_idx']}"
            
            print(f"{i:>3} | {seg_type:^8} | {id_str:>11} | {wp_range:>12} | {seg['distance']:>8.1f} | {seg['max_curvature']:>7.4f}")
        
        # 교차로 상세
        if info['junctions']:
            print(f"\n{'─'*70}")
            print(f" 교차로 상세 (NMPC 주의 구간)")
            print(f"{'─'*70}")
            for jct in info['junctions']:
                wp_start = info['waypoints_data'][jct['wp_range'][0]]
                wp_end = info['waypoints_data'][jct['wp_range'][1]]
                
                # 교차로 진입/이탈 각도 차이
                yaw_diff = wp_end['yaw'] - wp_start['yaw']
                yaw_diff = (yaw_diff + 180.0) % 360.0 - 180.0
                
                print(f"  Junction ID: {jct['junction_id']}")
                print(f"    위치: ({wp_start['x']:.1f}, {wp_start['y']:.1f}) → ({wp_end['x']:.1f}, {wp_end['y']:.1f})")
                print(f"    웨이포인트: {jct['wp_range'][0]}~{jct['wp_range'][1]} ({jct['wp_range'][1]-jct['wp_range'][0]+1}개)")
                print(f"    거리: {jct['distance']:.1f}m")
                print(f"    방향 변화: {yaw_diff:+.1f}°")
                print(f"    최대 곡률: {jct['max_curvature']:.4f} (1/m)")
                
                severity = "낮음"
                if abs(yaw_diff) > 60:
                    severity = "🔴 매우 높음 (90° 급선회 위험)"
                elif abs(yaw_diff) > 30:
                    severity = "🟡 높음"
                elif abs(yaw_diff) > 15:
                    severity = "🟢 보통"
                print(f"    NMPC 난이도: {severity}")
                print()

    # ─────────────────────────────────────────
    # Trajectory 저장/내보내기
    # ─────────────────────────────────────────
    def export_route(self, filepath=None):
        """현재 경로를 JSON으로 저장"""
        if self.current_route_info is None:
            print("Error: 먼저 경로를 생성하세요 (route 명령)")
            return
        
        if filepath is None:
            info = self.current_route_info
            filepath = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"scenario_s{info['start_idx']}_e{info['end_idx']}.json"
            )
        
        export_data = {
            'map': self.carla_map.name,
            'start_spawn_idx': self.current_route_info['start_idx'],
            'end_spawn_idx': self.current_route_info['end_idx'],
            'total_waypoints': self.current_route_info['total_waypoints'],
            'total_distance': round(self.current_route_info['total_distance'], 2),
            'num_junctions': len(self.current_route_info['junctions']),
            'junctions_summary': [
                {
                    'junction_id': j['junction_id'],
                    'wp_range': list(j['wp_range']),
                    'max_curvature': round(j['max_curvature'], 6),
                    'distance': round(j['distance'], 2)
                }
                for j in self.current_route_info['junctions']
            ],
            'waypoints': [
                {
                    'x': round(wp['x'], 3),
                    'y': round(wp['y'], 3),
                    'z': round(wp['z'], 3),
                    'yaw': round(wp['yaw'], 2),
                    'road_id': wp['road_id'],
                    'lane_id': wp['lane_id'],
                    'is_junction': wp['is_junction'],
                    'curvature': round(wp['curvature'], 6)
                }
                for wp in self.current_route_info['waypoints_data']
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"경로 저장 완료: {filepath}")
        return filepath

    # ─────────────────────────────────────────
    # 교차로 전체 탐색
    # ─────────────────────────────────────────
    def scan_junctions(self):
        """맵의 모든 교차로를 탐색하고 정보를 출력"""
        waypoints = self.carla_map.generate_waypoints(2.0)
        
        junction_map = {}
        for wp in waypoints:
            if wp.is_junction:
                jid = wp.junction_id
                if jid not in junction_map:
                    junction_map[jid] = {
                        'locations': [],
                        'wp_count': 0
                    }
                junction_map[jid]['locations'].append(wp.transform.location)
                junction_map[jid]['wp_count'] += 1
        
        print(f"\n{'='*60}")
        print(f" 교차로 스캔 결과 ({len(junction_map)}개)")
        print(f"{'='*60}")
        print(f"{'JCT ID':>7} | {'Center X':>9} | {'Center Y':>9} | {'WP Count':>8} | {'Radius':>7}")
        print(f"{'─'*7}-+-{'─'*9}-+-{'─'*9}-+-{'─'*8}-+-{'─'*7}")
        
        for jid, data in sorted(junction_map.items()):
            locs = data['locations']
            cx = sum(l.x for l in locs) / len(locs)
            cy = sum(l.y for l in locs) / len(locs)
            
            # 교차로 반경 추정
            max_r = max(math.sqrt((l.x-cx)**2 + (l.y-cy)**2) for l in locs)
            
            print(f"{jid:>7} | {cx:>9.1f} | {cy:>9.1f} | {data['wp_count']:>8} | {max_r:>7.1f}m")
            
            # 교차로 중심에 라벨 표시
            self.debug.draw_string(
                carla.Location(cx, cy, 5.0),
                f"J{jid}", draw_shadow=True, 
                color=COLOR_JUNCTION, life_time=DRAW_LIFE
            )
        
        return junction_map

    def focus_junction(self, junction_id, height=40.0):
        """특정 교차로로 카메라 이동"""
        waypoints = self.carla_map.generate_waypoints(2.0)
        locs = [wp.transform.location for wp in waypoints if wp.is_junction and wp.junction_id == junction_id]
        
        if not locs:
            print(f"Error: Junction ID {junction_id}를 찾을 수 없습니다.")
            return
        
        cx = sum(l.x for l in locs) / len(locs)
        cy = sum(l.y for l in locs) / len(locs)
        
        self.focus_location(carla.Location(cx, cy, 0), height)
        print(f"Junction {junction_id}로 이동: ({cx:.1f}, {cy:.1f})")

    # ─────────────────────────────────────────
    # 다중 경로 비교
    # ─────────────────────────────────────────
    def list_routes_through_junction(self, junction_id, max_routes=10):
        """특정 교차로를 통과하는 경로들을 탐색"""
        print(f"\nJunction {junction_id}을 통과하는 경로 탐색 중...")
        
        # 교차로 중심 좌표 계산
        waypoints = self.carla_map.generate_waypoints(2.0)
        jct_locs = [wp.transform.location for wp in waypoints if wp.is_junction and wp.junction_id == junction_id]
        
        if not jct_locs:
            print(f"Error: Junction {junction_id}를 찾을 수 없습니다.")
            return []
        
        jct_cx = sum(l.x for l in jct_locs) / len(jct_locs)
        jct_cy = sum(l.y for l in jct_locs) / len(jct_locs)
        jct_center = carla.Location(jct_cx, jct_cy, 0)
        
        # 교차로에 가까운 Spawn Point 쌍 탐색
        nearby_spawns = []
        for idx, sp in enumerate(self.spawn_points):
            dist = math.sqrt((sp.location.x - jct_cx)**2 + (sp.location.y - jct_cy)**2)
            if dist < 300.0:  # 300m 이내
                nearby_spawns.append((idx, dist))
        
        nearby_spawns.sort(key=lambda x: x[1])
        
        found_routes = []
        tested = set()
        
        for i, (start_idx, _) in enumerate(nearby_spawns):
            if len(found_routes) >= max_routes:
                break
            for j, (end_idx, _) in enumerate(nearby_spawns):
                if start_idx == end_idx:
                    continue
                pair = (start_idx, end_idx)
                if pair in tested:
                    continue
                tested.add(pair)
                
                try:
                    route = self.grp.trace_route(
                        self.spawn_points[start_idx].location,
                        self.spawn_points[end_idx].location
                    )
                    
                    # 교차로를 통과하는지 확인
                    passes_junction = any(
                        wp.is_junction and wp.junction_id == junction_id 
                        for wp, _ in route
                    )
                    
                    if passes_junction:
                        # 교차로 통과 각도 계산
                        jct_wps = [(i, wp) for i, (wp, _) in enumerate(route) 
                                   if wp.is_junction and wp.junction_id == junction_id]
                        
                        if len(jct_wps) >= 2:
                            entry_yaw = jct_wps[0][1].transform.rotation.yaw
                            exit_yaw = jct_wps[-1][1].transform.rotation.yaw
                            turn_angle = (exit_yaw - entry_yaw + 180.0) % 360.0 - 180.0
                        else:
                            turn_angle = 0.0
                        
                        found_routes.append({
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'total_wp': len(route),
                            'turn_angle': turn_angle,
                            'route': route
                        })
                        
                except Exception:
                    continue
        
        # 결과 출력
        print(f"\n{'='*60}")
        print(f" Junction {junction_id} 통과 경로 ({len(found_routes)}개)")
        print(f"{'='*60}")
        print(f"{'#':>3} | {'Start':>5} → {'End':>5} | {'WPs':>5} | {'Turn Angle':>12} | Type")
        print(f"{'─'*3}-+-{'─'*14}-+-{'─'*5}-+-{'─'*12}-+-{'─'*10}")
        
        for i, r in enumerate(found_routes):
            angle = r['turn_angle']
            if abs(angle) < 15:
                turn_type = "직진"
            elif angle > 0:
                turn_type = f"좌회전"
            else:
                turn_type = f"우회전"
            
            print(f"{i:>3} | S[{r['start_idx']:>3}] → S[{r['end_idx']:>3}] | {r['total_wp']:>5} | {angle:>+10.1f}° | {turn_type}")
        
        return found_routes


def print_help():
    """도움말 출력"""
    print(f"""
{'='*60}
 CARLA Route Inspector — 명령어 목록
{'='*60}

  시각화 명령:
    spawns           모든 Spawn Point 표시
    roads [간격]     도로 토폴로지 표시 (기본 5m)
    birdseye [높이]  조감도 설정 (기본 200m)
    
  경로 명령:
    route S E        Spawn[S] → Spawn[E] 경로 생성
    export [파일명]  현재 경로를 JSON으로 저장
    
  교차로 명령:
    junctions        모든 교차로 스캔
    focus J [높이]   Junction[J]로 카메라 이동
    jroutes J        Junction[J] 통과 경로 탐색
    
  기타:
    clear            화면의 드로잉 초기화
    help             이 도움말 표시
    quit             종료
{'='*60}
""")


def main():
    parser = argparse.ArgumentParser(description='CARLA Route Inspector')
    parser.add_argument('--map', type=str, default=None, help='CARLA 맵 이름 (예: Town04)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='CARLA 서버 호스트')
    parser.add_argument('--port', type=int, default=2000, help='CARLA 서버 포트')
    args = parser.parse_args()
    
    print("CARLA Route Inspector 시작...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    
    inspector = RouteInspector(client, map_name=args.map)
    
    # 초기 설정: 조감도 + spawn points
    inspector.set_bird_eye_view()
    inspector.draw_spawn_points()
    
    print_help()
    
    while True:
        try:
            cmd = input("\n[Inspector] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break
        
        if not cmd:
            continue
        
        parts = cmd.split()
        action = parts[0].lower()
        
        try:
            if action == 'quit' or action == 'exit' or action == 'q':
                print("종료합니다.")
                break
            
            elif action == 'help' or action == 'h':
                print_help()
            
            elif action == 'spawns':
                inspector.draw_spawn_points()
            
            elif action == 'roads':
                spacing = float(parts[1]) if len(parts) > 1 else 5.0
                inspector.draw_road_topology(spacing)
            
            elif action == 'birdseye' or action == 'bird':
                height = float(parts[1]) if len(parts) > 1 else 200.0
                inspector.set_bird_eye_view(height=height)
            
            elif action == 'route':
                if len(parts) < 3:
                    print("사용법: route <start_idx> <end_idx>")
                    continue
                start_idx = int(parts[1])
                end_idx = int(parts[2])
                inspector.generate_route(start_idx, end_idx)
            
            elif action == 'export':
                filepath = parts[1] if len(parts) > 1 else None
                inspector.export_route(filepath)
            
            elif action == 'junctions' or action == 'jct':
                inspector.scan_junctions()
            
            elif action == 'focus':
                if len(parts) < 2:
                    print("사용법: focus <junction_id> [높이]")
                    continue
                jid = int(parts[1])
                height = float(parts[2]) if len(parts) > 2 else 40.0
                inspector.focus_junction(jid, height)
            
            elif action == 'jroutes':
                if len(parts) < 2:
                    print("사용법: jroutes <junction_id>")
                    continue
                jid = int(parts[1])
                routes = inspector.list_routes_through_junction(jid)
                
                # 추가: 특정 경로 시각화 선택
                if routes:
                    sel = input("  시각화할 경로 번호 (Enter=건너뛰기): ").strip()
                    if sel.isdigit() and int(sel) < len(routes):
                        r = routes[int(sel)]
                        inspector.current_route = r['route']
                        inspector.current_route_info = inspector._analyze_route(
                            r['route'], r['start_idx'], r['end_idx']
                        )
                        inspector._draw_route(r['route'])
                        inspector._print_route_info(inspector.current_route_info)
            
            elif action == 'clear':
                # 드로잉을 지울 수 없으므로 life_time이 만료되길 기다려야 합니다
                print("드로잉은 60초 후 자동 소멸됩니다.")
                print("즉시 초기화하려면 CARLA를 재시작하세요.")
            
            else:
                print(f"알 수 없는 명령: {action}  (help로 명령어 목록 확인)")
                
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
