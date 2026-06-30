#!/usr/bin/env python
"""
NMPC Stress Test — 고곡률 교차로 집중 자동 테스트

고위험 교차로(방향변화 > 60°)를 통과하는 경로를 자동으로 탐색하고
무작위로 선택하여 NMPC 경로 추종을 반복 실행합니다.

사용법:
    # 기본: Town04에서 20회 스트레스 테스트
    python nmpc_stress_test.py

    # 맵과 횟수 지정
    python nmpc_stress_test.py --map Town04 --runs 50

    # 특정 교차로 집중 테스트
    python nmpc_stress_test.py --junction 255

    # 최소 방향변화 각도 설정 (기본 60°)
    python nmpc_stress_test.py --min-angle 45

    # 결과를 JSON으로 저장
    python nmpc_stress_test.py --output stress_results.json
"""

import sys
import os
import json
import math
import time
import random
import argparse
from datetime import datetime
from collections import defaultdict

try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'carla'))
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Runner의 run_scenario를 재사용
from nmpc_carla_runner import run_scenario


class JunctionScanner:
    """맵의 교차로를 스캔하고 고위험 경로를 탐색합니다"""
    
    def __init__(self, world):
        self.world = world
        self.carla_map = world.get_map()
        self.grp = GlobalRoutePlanner(self.carla_map, 2.0)
        self.spawn_points = self.carla_map.get_spawn_points()
    
    def scan_junctions(self):
        """모든 교차로의 위치와 크기를 스캔"""
        waypoints = self.carla_map.generate_waypoints(2.0)
        
        junction_map = {}
        for wp in waypoints:
            if wp.is_junction:
                jid = wp.junction_id
                if jid not in junction_map:
                    junction_map[jid] = []
                junction_map[jid].append(wp.transform.location)
        
        result = {}
        for jid, locs in junction_map.items():
            cx = sum(l.x for l in locs) / len(locs)
            cy = sum(l.y for l in locs) / len(locs)
            max_r = max(math.sqrt((l.x-cx)**2 + (l.y-cy)**2) for l in locs)
            result[jid] = {
                'center_x': cx, 'center_y': cy,
                'radius': max_r, 'wp_count': len(locs)
            }
        
        return result

    def find_stress_routes(self, min_turn_angle=60.0, target_junction_id=None, 
                           max_routes=100, max_search=500):
        """
        고위험 교차로를 통과하는 경로들을 탐색합니다.
        
        Args:
            min_turn_angle: 최소 방향 변화 각도 (기본 60°)
            target_junction_id: 특정 교차로만 타겟 (None이면 전체)
            max_routes: 최대 경로 수
            max_search: 최대 탐색 시도 횟수
        
        Returns:
            list of route dicts with metadata
        """
        junctions = self.scan_junctions()
        spawn_count = len(self.spawn_points)
        
        print(f"\n{'='*60}")
        print(f" 고위험 경로 탐색 (min_angle={min_turn_angle}°)")
        print(f"{'='*60}")
        print(f"  맵: {self.carla_map.name}")
        print(f"  Spawn Points: {spawn_count}개")
        print(f"  교차로: {len(junctions)}개")
        
        found_routes = []
        tested_pairs = set()
        search_count = 0
        
        while len(found_routes) < max_routes and search_count < max_search:
            search_count += 1
            
            # 랜덤 start/end 선택
            s_idx = random.randint(0, spawn_count - 1)
            e_idx = random.randint(0, spawn_count - 1)
            if s_idx == e_idx:
                continue
            
            pair = (s_idx, e_idx)
            if pair in tested_pairs:
                continue
            tested_pairs.add(pair)
            
            try:
                route = self.grp.trace_route(
                    self.spawn_points[s_idx].location,
                    self.spawn_points[e_idx].location
                )
            except Exception:
                continue
            
            if len(route) < 10:
                continue
            
            # 교차로 통과 분석
            route_junctions = self._analyze_route_junctions(route)
            
            # 필터: 고위험 교차로가 있는 경로만 선택
            hard_junctions = []
            for jct in route_junctions:
                if abs(jct['turn_angle']) >= min_turn_angle:
                    if target_junction_id is not None:
                        if jct['junction_id'] == target_junction_id:
                            hard_junctions.append(jct)
                    else:
                        hard_junctions.append(jct)
            
            if not hard_junctions:
                continue
            
            # 경로 총 거리 계산
            total_dist = 0.0
            for i in range(1, len(route)):
                l0 = route[i-1][0].transform.location
                l1 = route[i][0].transform.location
                total_dist += math.sqrt((l1.x-l0.x)**2 + (l1.y-l0.y)**2)
            
            route_info = {
                'start_idx': s_idx,
                'end_idx': e_idx,
                'total_wp': len(route),
                'total_dist': round(total_dist, 1),
                'hard_junctions': hard_junctions,
                'max_turn_angle': max(abs(j['turn_angle']) for j in hard_junctions),
                'num_hard_junctions': len(hard_junctions)
            }
            
            found_routes.append(route_info)
            
            if len(found_routes) % 5 == 0:
                print(f"  발견: {len(found_routes)}개 / 탐색: {search_count}회")
        
        # 난이도(최대 회전각) 내림차순 정렬
        found_routes.sort(key=lambda r: r['max_turn_angle'], reverse=True)
        
        print(f"\n  탐색 완료: {len(found_routes)}개 고위험 경로 발견 (시도: {search_count}회)")
        
        # 요약 출력
        if found_routes:
            print(f"\n{'─'*70}")
            print(f"{'#':>3} | {'S→E':>10} | {'WPs':>4} | {'Dist':>7} | {'Hard JCTs':>9} | {'MaxAngle':>9}")
            print(f"{'─'*3}-+-{'─'*10}-+-{'─'*4}-+-{'─'*7}-+-{'─'*9}-+-{'─'*9}")
            
            for i, r in enumerate(found_routes[:20]):  # 상위 20개만 출력
                pair = f"S[{r['start_idx']:>3}]→S[{r['end_idx']:>3}]"
                print(f"{i:>3} | {pair} | {r['total_wp']:>4} | {r['total_dist']:>6.0f}m | {r['num_hard_junctions']:>9} | {r['max_turn_angle']:>+8.1f}°")
        
        return found_routes
    
    def _analyze_route_junctions(self, route):
        """경로의 교차로 통과 정보를 분석"""
        junctions = []
        
        in_junction = False
        jct_entry_idx = 0
        jct_entry_yaw = 0.0
        current_jct_id = None
        jct_max_curvature = 0.0
        
        for i, (wp, _) in enumerate(route):
            if wp.is_junction and not in_junction:
                # 교차로 진입
                in_junction = True
                jct_entry_idx = i
                jct_entry_yaw = wp.transform.rotation.yaw
                current_jct_id = wp.junction_id
                jct_max_curvature = 0.0
            
            elif wp.is_junction and in_junction and wp.junction_id != current_jct_id:
                # 다른 교차로로 진입 (이전 교차로 종료)
                exit_yaw = route[i-1][0].transform.rotation.yaw
                turn_angle = (exit_yaw - jct_entry_yaw + 180.0) % 360.0 - 180.0
                
                junctions.append({
                    'junction_id': current_jct_id,
                    'entry_idx': jct_entry_idx,
                    'exit_idx': i - 1,
                    'turn_angle': round(turn_angle, 1),
                    'max_curvature': round(jct_max_curvature, 6)
                })
                
                # 새 교차로 시작
                jct_entry_idx = i
                jct_entry_yaw = wp.transform.rotation.yaw
                current_jct_id = wp.junction_id
                jct_max_curvature = 0.0
            
            elif not wp.is_junction and in_junction:
                # 교차로 이탈
                exit_yaw = wp.transform.rotation.yaw
                turn_angle = (exit_yaw - jct_entry_yaw + 180.0) % 360.0 - 180.0
                
                # 곡률 계산
                if i >= 2:
                    p0 = route[i-2][0].transform.location
                    p1 = route[i-1][0].transform.location
                    p2 = wp.transform.location
                    ax, ay = p1.x-p0.x, p1.y-p0.y
                    bx, by = p2.x-p1.x, p2.y-p1.y
                    cross = abs(ax*by - ay*bx)
                    a = math.sqrt(ax*ax+ay*ay)
                    b = math.sqrt(bx*bx+by*by)
                    cx_, cy_ = p2.x-p0.x, p2.y-p0.y
                    c = math.sqrt(cx_*cx_+cy_*cy_)
                    denom = a*b*c
                    if denom > 1e-6:
                        jct_max_curvature = max(jct_max_curvature, 2.0*cross/denom)
                
                junctions.append({
                    'junction_id': current_jct_id,
                    'entry_idx': jct_entry_idx,
                    'exit_idx': i - 1,
                    'turn_angle': round(turn_angle, 1),
                    'max_curvature': round(jct_max_curvature, 6)
                })
                in_junction = False
                current_jct_id = None
        
        return junctions


def run_stress_test(client, world, routes, target_speed=50.0, max_ticks_per_run=20000):
    """
    발견된 고위험 경로들을 순차적으로 실행하고 결과를 수집합니다.
    """
    total = len(routes)
    results = []
    
    print(f"\n{'='*70}")
    print(f" NMPC Stress Test 시작: {total}개 경로")
    print(f" 목표 속도: {target_speed} km/h, 최대 틱/경로: {max_ticks_per_run}")
    print(f"{'='*70}\n")
    
    for i, route_info in enumerate(routes):
        s_idx = route_info['start_idx']
        e_idx = route_info['end_idx']
        
        print(f"\n{'━'*70}")
        print(f" [{i+1}/{total}] Spawn[{s_idx}] → Spawn[{e_idx}]")
        print(f" 난이도: max_angle={route_info['max_turn_angle']:+.1f}°, "
              f"hard_junctions={route_info['num_hard_junctions']}")
        print(f"{'━'*70}")
        
        try:
            metrics = run_scenario(
                client, world,
                start_idx=s_idx,
                end_idx=e_idx,
                target_speed=target_speed,
                max_ticks=max_ticks_per_run,
                collect_metrics=True
            )
            
            if metrics is None:
                metrics = {'completed': False, 'error': 'No metrics returned'}
            
            metrics['route_idx'] = i
            metrics['start_spawn'] = s_idx
            metrics['end_spawn'] = e_idx
            metrics['max_turn_angle'] = route_info['max_turn_angle']
            metrics['num_hard_junctions'] = route_info['num_hard_junctions']
            metrics['total_dist'] = route_info['total_dist']
            
            results.append(metrics)
            
            # 중간 결과 출력
            status = "✅ PASS" if metrics.get('completed', False) else "❌ FAIL"
            print(f"\n  결과: {status}")
            print(f"  Fallback: {metrics.get('fallback_count', 'N/A')}회, "
                  f"MaxKKT: {metrics.get('max_kkt_error', 'N/A')}, "
                  f"MaxSteerRate: {metrics.get('max_steer_rate', 'N/A')}")
        
        except KeyboardInterrupt:
            print("\n\n사용자에 의한 스트레스 테스트 중단.")
            break
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            results.append({
                'route_idx': i,
                'start_spawn': s_idx,
                'end_spawn': e_idx,
                'completed': False,
                'error': str(e),
                'max_turn_angle': route_info['max_turn_angle']
            })
        
        # 시뮬레이터 안정화 (차량 파괴 후)
        time.sleep(1.0)
    
    return results


def print_summary(results):
    """전체 스트레스 테스트 결과 요약"""
    total = len(results)
    if total == 0:
        print("결과 없음")
        return
    
    completed = sum(1 for r in results if r.get('completed', False))
    failed = total - completed
    collisions = sum(1 for r in results if r.get('collision', False))
    
    avg_fallback = 0
    max_fallback = 0
    fallback_counts = [r.get('fallback_count', 0) for r in results if 'fallback_count' in r]
    if fallback_counts:
        avg_fallback = sum(fallback_counts) / len(fallback_counts)
        max_fallback = max(fallback_counts)
    
    print(f"\n{'='*70}")
    print(f" NMPC Stress Test 최종 결과")
    print(f"{'='*70}")
    print(f"  총 실행: {total}회")
    print(f"  ✅ 완주: {completed}회 ({completed/total*100:.0f}%)")
    print(f"  ❌ 실패: {failed}회 ({failed/total*100:.0f}%)")
    print(f"  💥 충돌: {collisions}회")
    print(f"  Fallback 평균/최대: {avg_fallback:.1f} / {max_fallback}")
    
    # 실패 케이스 상세
    if failed > 0:
        print(f"\n{'─'*70}")
        print(f" 실패 케이스 상세")
        print(f"{'─'*70}")
        for r in results:
            if not r.get('completed', False):
                angle = r.get('max_turn_angle', 0)
                err = r.get('error', r.get('max_kkt_error', 'N/A'))
                print(f"  S[{r['start_spawn']:>3}]→S[{r['end_spawn']:>3}] | "
                      f"angle={angle:+.1f}° | error={err}")
    
    # 난이도별 통계
    print(f"\n{'─'*70}")
    print(f" 난이도별 성공률")
    print(f"{'─'*70}")
    
    buckets = {'60-70°': [], '70-80°': [], '80-90°': [], '90°+': []}
    for r in results:
        a = abs(r.get('max_turn_angle', 0))
        if a >= 90:
            buckets['90°+'].append(r)
        elif a >= 80:
            buckets['80-90°'].append(r)
        elif a >= 70:
            buckets['70-80°'].append(r)
        else:
            buckets['60-70°'].append(r)
    
    for label, items in buckets.items():
        if items:
            passed = sum(1 for i in items if i.get('completed', False))
            print(f"  {label:>8}: {passed}/{len(items)} 완주 ({passed/len(items)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description='NMPC Stress Test')
    parser.add_argument('--map', type=str, default=None,
                        help='CARLA 맵 이름 (예: Town04)')
    parser.add_argument('--runs', type=int, default=20,
                        help='실행 횟수 (기본 20)')
    parser.add_argument('--junction', type=int, default=None,
                        help='특정 교차로 ID만 타겟')
    parser.add_argument('--min-angle', type=float, default=60.0,
                        help='최소 방향변화 각도 (기본 60°)')
    parser.add_argument('--speed', type=float, default=50.0,
                        help='목표 속도 (km/h, 기본 50)')
    parser.add_argument('--max-ticks', type=int, default=20000,
                        help='경로당 최대 틱 수 (기본 20000)')
    parser.add_argument('--output', type=str, default=None,
                        help='결과 JSON 저장 경로')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    args = parser.parse_args()
    
    print("NMPC Stress Test 시작...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    
    if args.map:
        print(f"맵 로딩: {args.map}")
        client.load_world(args.map)
        time.sleep(2.0)
    
    world = client.get_world()
    
    # Phase 1: 고위험 경로 탐색
    scanner = JunctionScanner(world)
    stress_routes = scanner.find_stress_routes(
        min_turn_angle=args.min_angle,
        target_junction_id=args.junction,
        max_routes=args.runs * 3,  # 여유분 확보
        max_search=args.runs * 20
    )
    
    if not stress_routes:
        print("고위험 경로를 찾을 수 없습니다. --min-angle을 낮추거나 다른 맵을 시도하세요.")
        return
    
    # 실행할 경로 선택 (최대 runs개, 랜덤 셔플)
    selected = stress_routes[:min(args.runs, len(stress_routes))]
    random.shuffle(selected)
    
    print(f"\n{len(selected)}개 경로 선택됨. 스트레스 테스트 시작합니다.")
    input("Enter를 누르면 시작합니다... (Ctrl+C로 중단)")
    
    # Phase 2: 스트레스 테스트 실행
    results = run_stress_test(
        client, world, selected,
        target_speed=args.speed,
        max_ticks_per_run=args.max_ticks
    )
    
    # Phase 3: 결과 요약
    print_summary(results)
    
    # Phase 4: 결과 저장
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'map': world.get_map().name,
            'config': {
                'target_speed_kmh': args.speed,
                'min_angle': args.min_angle,
                'max_ticks': args.max_ticks,
                'target_junction': args.junction
            },
            'summary': {
                'total': len(results),
                'completed': sum(1 for r in results if r.get('completed')),
                'failed': sum(1 for r in results if not r.get('completed'))
            },
            'results': results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n결과 저장: {args.output}")


if __name__ == '__main__':
    main()
