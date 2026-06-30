#!/usr/bin/env python
"""
NMPC Stress Test 결과 분석기

nmpc_stress_test.py가 출력한 JSON 결과 파일을 분석합니다.
단일 파일 분석 또는 여러 파일 비교(수정 전/후)를 지원합니다.

사용법:
    # 단일 결과 분석
    python analyze_stress_results.py results.json

    # 여러 결과 비교 (수정 전 vs 후)
    python analyze_stress_results.py before.json after.json

    # 실패 케이스만 상세 출력
    python analyze_stress_results.py results.json --failures

    # matplotlib 차트 생성
    python analyze_stress_results.py results.json --plot
"""

import sys
import os
import json
import math
import argparse
from collections import defaultdict


def load_result(filepath):
    """결과 JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def safe_get(d, key, default=0):
    """dict에서 안전하게 값을 가져옴 (None도 default로 처리)"""
    v = d.get(key, default)
    return v if v is not None else default


# ─────────────────────────────────────────────────────────────
# 1. 기본 통계 분석
# ─────────────────────────────────────────────────────────────

def analyze_basic(data, label=""):
    """기본 통계를 분석하여 출력"""
    results = data.get('results', [])
    config = data.get('config', {})
    total = len(results)
    
    if total == 0:
        print("결과가 비어 있습니다.")
        return {}
    
    completed = [r for r in results if r.get('completed', False)]
    failed = [r for r in results if not r.get('completed', False)]
    
    # 메트릭 수집
    fallbacks = [safe_get(r, 'fallback_count') for r in results if 'fallback_count' in r]
    kkt_errors = [safe_get(r, 'max_kkt_error') for r in results if 'max_kkt_error' in r]
    steer_rates = [safe_get(r, 'max_steer_rate') for r in results if 'max_steer_rate' in r]
    max_accels = [safe_get(r, 'max_accel') for r in results if 'max_accel' in r]
    avg_speeds = [safe_get(r, 'avg_speed_ms') for r in results if 'avg_speed_ms' in r]
    elapsed_times = [safe_get(r, 'elapsed_sec') for r in results if 'elapsed_sec' in r]
    
    title = f" 분석 결과: {label}" if label else " 분석 결과"
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"  파일: {label}")
    print(f"  맵: {data.get('map', 'N/A')}")
    print(f"  시간: {data.get('timestamp', 'N/A')}")
    print(f"  설정: speed={config.get('target_speed_kmh', 'N/A')} km/h, "
          f"min_angle={config.get('min_angle', 'N/A')}°")
    
    print(f"\n  {'─'*50}")
    print(f"  총 실행: {total}")
    print(f"  ✅ 완주: {len(completed)} ({len(completed)/total*100:.1f}%)")
    print(f"  ❌ 실패: {len(failed)} ({len(failed)/total*100:.1f}%)")
    
    stats = {
        'total': total,
        'completed': len(completed),
        'failed': len(failed),
        'pass_rate': len(completed) / total * 100
    }
    
    if fallbacks:
        stats['fallback_avg'] = sum(fallbacks) / len(fallbacks)
        stats['fallback_max'] = max(fallbacks)
        stats['fallback_total'] = sum(fallbacks)
        print(f"\n  Fallback 횟수:")
        print(f"    합계: {stats['fallback_total']}")
        print(f"    평균: {stats['fallback_avg']:.1f}")
        print(f"    최대: {stats['fallback_max']}")
    
    if kkt_errors:
        stats['kkt_avg'] = sum(kkt_errors) / len(kkt_errors)
        stats['kkt_max'] = max(kkt_errors)
        print(f"\n  KKT Error:")
        print(f"    평균: {stats['kkt_avg']:.4f}")
        print(f"    최대: {stats['kkt_max']:.4f}")
    
    if steer_rates:
        stats['steer_rate_avg'] = sum(steer_rates) / len(steer_rates)
        stats['steer_rate_max'] = max(steer_rates)
        print(f"\n  Steer Rate (rad/step):")
        print(f"    평균: {stats['steer_rate_avg']:.4f}")
        print(f"    최대: {stats['steer_rate_max']:.4f}")
    
    if max_accels:
        stats['accel_avg'] = sum(max_accels) / len(max_accels)
        stats['accel_max'] = max(max_accels)
        print(f"\n  Max Accel (m/s²):")
        print(f"    평균: {stats['accel_avg']:.2f}")
        print(f"    최대: {stats['accel_max']:.2f}")
    
    if avg_speeds:
        stats['speed_avg'] = sum(avg_speeds) / len(avg_speeds)
        print(f"\n  평균 속도: {stats['speed_avg']:.1f} m/s ({stats['speed_avg']*3.6:.1f} km/h)")
    
    if elapsed_times:
        stats['time_avg'] = sum(elapsed_times) / len(elapsed_times)
        stats['time_total'] = sum(elapsed_times)
        print(f"\n  실행 시간:")
        print(f"    평균: {stats['time_avg']:.1f}초")
        print(f"    합계: {stats['time_total']:.0f}초 ({stats['time_total']/60:.1f}분)")
    
    return stats


# ─────────────────────────────────────────────────────────────
# 2. 난이도별 분석
# ─────────────────────────────────────────────────────────────

def analyze_by_difficulty(data, label=""):
    """방향변화 각도(난이도)별 성공률 분석"""
    results = data.get('results', [])
    
    # 10° 구간 버킷팅
    buckets = defaultdict(lambda: {'pass': 0, 'fail': 0, 'fallbacks': [], 'kkt': []})
    
    for r in results:
        angle = abs(safe_get(r, 'max_turn_angle'))
        bucket = f"{int(angle // 10) * 10}-{int(angle // 10) * 10 + 10}°"
        
        if r.get('completed', False):
            buckets[bucket]['pass'] += 1
        else:
            buckets[bucket]['fail'] += 1
        
        if 'fallback_count' in r:
            buckets[bucket]['fallbacks'].append(safe_get(r, 'fallback_count'))
        if 'max_kkt_error' in r:
            buckets[bucket]['kkt'].append(safe_get(r, 'max_kkt_error'))
    
    print(f"\n{'─'*70}")
    print(f" 난이도별 분석 {f'({label})' if label else ''}")
    print(f"{'─'*70}")
    print(f"{'Angle':>10} | {'Total':>5} | {'Pass':>5} | {'Fail':>5} | {'Rate':>6} | {'AvgFB':>6} | {'AvgKKT':>8}")
    print(f"{'─'*10}-+-{'─'*5}-+-{'─'*5}-+-{'─'*5}-+-{'─'*6}-+-{'─'*6}-+-{'─'*8}")
    
    for bucket in sorted(buckets.keys(), key=lambda x: int(x.split('-')[0])):
        b = buckets[bucket]
        total = b['pass'] + b['fail']
        rate = b['pass'] / total * 100 if total > 0 else 0
        avg_fb = sum(b['fallbacks']) / len(b['fallbacks']) if b['fallbacks'] else 0
        avg_kkt = sum(b['kkt']) / len(b['kkt']) if b['kkt'] else 0
        
        # 색상 코드 (터미널)
        if rate >= 90:
            indicator = "🟢"
        elif rate >= 70:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f"{bucket:>10} | {total:>5} | {b['pass']:>5} | {b['fail']:>5} | "
              f"{rate:>5.1f}% | {avg_fb:>6.1f} | {avg_kkt:>8.4f} {indicator}")
    
    return dict(buckets)


# ─────────────────────────────────────────────────────────────
# 3. 실패 케이스 상세 분석
# ─────────────────────────────────────────────────────────────

def analyze_failures(data, label=""):
    """실패 케이스만 상세 분석"""
    results = data.get('results', [])
    failed = [r for r in results if not r.get('completed', False)]
    
    if not failed:
        print(f"\n✅ 실패 케이스 없음 {f'({label})' if label else ''}")
        return
    
    print(f"\n{'─'*70}")
    print(f" 실패 케이스 상세 ({len(failed)}건) {f'({label})' if label else ''}")
    print(f"{'─'*70}")
    
    # 에러 유형 분류
    error_types = defaultdict(list)
    
    for r in failed:
        error = r.get('error', 'Unknown')
        if 'error' not in r:
            # 타임아웃 또는 미완주
            if safe_get(r, 'sim_ticks', 0) >= 19000:
                error = 'TIMEOUT'
            else:
                error = 'INCOMPLETE'
        error_types[error].append(r)
    
    print(f"\n  에러 유형 분포:")
    for err_type, cases in sorted(error_types.items(), key=lambda x: -len(x[1])):
        print(f"    {err_type}: {len(cases)}건")
    
    print(f"\n{'#':>3} | {'Route':>14} | {'Angle':>7} | {'Ticks':>7} | {'FB':>4} | {'KKT':>8} | Error")
    print(f"{'─'*3}-+-{'─'*14}-+-{'─'*7}-+-{'─'*7}-+-{'─'*4}-+-{'─'*8}-+-{'─'*20}")
    
    for i, r in enumerate(failed):
        route = f"S[{safe_get(r, 'start_spawn'):>3}]→S[{safe_get(r, 'end_spawn'):>3}]"
        angle = safe_get(r, 'max_turn_angle')
        ticks = safe_get(r, 'sim_ticks')
        fb = safe_get(r, 'fallback_count')
        kkt = safe_get(r, 'max_kkt_error')
        error = r.get('error', 'N/A')
        if len(error) > 20:
            error = error[:17] + "..."
        
        print(f"{i:>3} | {route} | {angle:>+6.1f}° | {ticks:>7} | {fb:>4} | {kkt:>8.4f} | {error}")
    
    # 실패 패턴 분석
    fail_angles = [abs(safe_get(r, 'max_turn_angle')) for r in failed]
    pass_angles = [abs(safe_get(r, 'max_turn_angle')) for r in results if r.get('completed', False)]
    
    if fail_angles and pass_angles:
        print(f"\n  실패 경로 평균 각도: {sum(fail_angles)/len(fail_angles):.1f}°")
        print(f"  성공 경로 평균 각도: {sum(pass_angles)/len(pass_angles):.1f}°")
        
        # 임계 각도 추정
        all_sorted = sorted(results, key=lambda r: abs(safe_get(r, 'max_turn_angle')))
        for r in all_sorted:
            if not r.get('completed', False):
                print(f"  최소 실패 각도: {abs(safe_get(r, 'max_turn_angle')):.1f}° "
                      f"(S[{safe_get(r, 'start_spawn')}]→S[{safe_get(r, 'end_spawn')}])")
                break


# ─────────────────────────────────────────────────────────────
# 4. 다중 파일 비교 (Before/After)
# ─────────────────────────────────────────────────────────────

def compare_results(data_list, labels):
    """여러 결과 파일을 비교 분석"""
    print(f"\n{'='*70}")
    print(f" 비교 분석: {' vs '.join(labels)}")
    print(f"{'='*70}")
    
    stats_list = []
    for data, label in zip(data_list, labels):
        stats = analyze_basic(data, label)
        stats_list.append(stats)
    
    # 비교 테이블
    print(f"\n{'─'*70}")
    print(f" 핵심 지표 비교")
    print(f"{'─'*70}")
    
    metrics = [
        ('완주율 (%)', 'pass_rate', '.1f'),
        ('Fallback 평균', 'fallback_avg', '.1f'),
        ('Fallback 최대', 'fallback_max', '.0f'),
        ('KKT Error 평균', 'kkt_avg', '.4f'),
        ('KKT Error 최대', 'kkt_max', '.4f'),
        ('Steer Rate 평균', 'steer_rate_avg', '.4f'),
        ('Steer Rate 최대', 'steer_rate_max', '.4f'),
        ('Max Accel 평균', 'accel_avg', '.2f'),
        ('평균 속도 (m/s)', 'speed_avg', '.1f'),
    ]
    
    # 헤더
    header = f"{'Metric':>20}"
    for label in labels:
        header += f" | {label:>12}"
    if len(labels) == 2:
        header += f" | {'Delta':>10}"
    print(header)
    
    sep = f"{'─'*20}"
    for _ in labels:
        sep += f"-+-{'─'*12}"
    if len(labels) == 2:
        sep += f"-+-{'─'*10}"
    print(sep)
    
    for metric_name, metric_key, fmt in metrics:
        row = f"{metric_name:>20}"
        values = []
        for stats in stats_list:
            val = stats.get(metric_key)
            if val is not None:
                row += f" | {val:>12{fmt}}"
                values.append(val)
            else:
                row += f" | {'N/A':>12}"
                values.append(None)
        
        if len(labels) == 2 and all(v is not None for v in values):
            delta = values[1] - values[0]
            sign = "+" if delta > 0 else ""
            # 완주율, 속도는 높을수록 좋고, 나머지는 낮을수록 좋음
            good_if_up = metric_key in ('pass_rate', 'speed_avg')
            is_better = (delta > 0) if good_if_up else (delta < 0)
            indicator = "✅" if is_better else ("⚠️" if delta != 0 else "→")
            row += f" | {sign}{delta:>8{fmt}} {indicator}"
        
        print(row)
    
    # 난이도별 비교
    print(f"\n{'─'*70}")
    print(f" 난이도별 완주율 비교")
    print(f"{'─'*70}")
    
    for data, label in zip(data_list, labels):
        analyze_by_difficulty(data, label)


# ─────────────────────────────────────────────────────────────
# 5. Matplotlib 차트 생성 (선택사항)
# ─────────────────────────────────────────────────────────────

def plot_results(data_list, labels, output_dir=None):
    """결과를 차트로 시각화 (matplotlib 필요)"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib이 설치되어 있지 않습니다. --plot 옵션을 사용하려면:")
        print("  pip install matplotlib")
        return
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NMPC Stress Test Analysis', fontsize=14, fontweight='bold')
    
    for data, label in zip(data_list, labels):
        results = data.get('results', [])
        
        angles = [abs(safe_get(r, 'max_turn_angle')) for r in results]
        completed = [r.get('completed', False) for r in results]
        fallbacks = [safe_get(r, 'fallback_count') for r in results]
        kkt_errors = [safe_get(r, 'max_kkt_error') for r in results]
        steer_rates = [safe_get(r, 'max_steer_rate') for r in results]
        
        # 1. 각도 vs 성공/실패 산점도
        ax = axes[0, 0]
        pass_angles = [a for a, c in zip(angles, completed) if c]
        fail_angles = [a for a, c in zip(angles, completed) if not c]
        ax.scatter(pass_angles, [1]*len(pass_angles), alpha=0.6, s=40, label=f'{label} Pass', marker='o')
        ax.scatter(fail_angles, [0]*len(fail_angles), alpha=0.6, s=40, label=f'{label} Fail', marker='x')
        ax.set_xlabel('Max Turn Angle (°)')
        ax.set_ylabel('Result')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Fail', 'Pass'])
        ax.set_title('Pass/Fail by Turn Angle')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. 각도 vs Fallback 횟수
        ax = axes[0, 1]
        ax.scatter(angles, fallbacks, alpha=0.5, s=30, label=label)
        ax.set_xlabel('Max Turn Angle (°)')
        ax.set_ylabel('Fallback Count')
        ax.set_title('Fallback Count by Turn Angle')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. 각도 vs KKT Error
        ax = axes[1, 0]
        ax.scatter(angles, kkt_errors, alpha=0.5, s=30, label=label)
        ax.set_xlabel('Max Turn Angle (°)')
        ax.set_ylabel('Max KKT Error')
        ax.set_title('KKT Error by Turn Angle')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 4. 난이도별 완주율 막대 그래프
        ax = axes[1, 1]
        buckets = defaultdict(lambda: {'pass': 0, 'fail': 0})
        for a, c in zip(angles, completed):
            b = f"{int(a // 10) * 10}-{int(a // 10) * 10 + 10}"
            if c:
                buckets[b]['pass'] += 1
            else:
                buckets[b]['fail'] += 1
        
        sorted_keys = sorted(buckets.keys(), key=lambda x: int(x.split('-')[0]))
        rates = []
        for k in sorted_keys:
            t = buckets[k]['pass'] + buckets[k]['fail']
            rates.append(buckets[k]['pass'] / t * 100 if t > 0 else 0)
        
        x_pos = range(len(sorted_keys))
        width = 0.35
        offset = labels.index(label) * width - width / 2 if len(labels) > 1 else 0
        bars = ax.bar([x + offset for x in x_pos], rates, width, label=label, alpha=0.7)
        
        # 막대 위에 수치 표시
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Turn Angle Range (°)')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_title('Pass Rate by Difficulty')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_keys, rotation=45, fontsize=8)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, 'stress_test_analysis.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n차트 저장: {chart_path}")


def main():
    parser = argparse.ArgumentParser(
        description='NMPC Stress Test 결과 분석기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  단일 분석:   python analyze_stress_results.py results.json
  비교 분석:   python analyze_stress_results.py before.json after.json
  실패 상세:   python analyze_stress_results.py results.json --failures
  차트 생성:   python analyze_stress_results.py results.json --plot
  전체 분석:   python analyze_stress_results.py results.json --failures --plot
        """
    )
    parser.add_argument('files', nargs='+', help='결과 JSON 파일 (1개: 단일 분석, 2개+: 비교)')
    parser.add_argument('--failures', action='store_true', help='실패 케이스 상세 출력')
    parser.add_argument('--plot', action='store_true', help='matplotlib 차트 생성')
    parser.add_argument('--labels', nargs='*', default=None,
                        help='비교 시 라벨 (예: --labels "수정 전" "수정 후")')
    args = parser.parse_args()
    
    # 파일 로드
    data_list = []
    labels = args.labels or []
    
    for i, filepath in enumerate(args.files):
        if not os.path.exists(filepath):
            print(f"Error: 파일을 찾을 수 없습니다: {filepath}")
            sys.exit(1)
        
        data = load_result(filepath)
        data_list.append(data)
        
        if i >= len(labels):
            # 자동 라벨: 파일명 사용
            labels.append(os.path.splitext(os.path.basename(filepath))[0])
    
    # 분석 실행
    if len(data_list) == 1:
        # 단일 파일 분석
        analyze_basic(data_list[0], labels[0])
        analyze_by_difficulty(data_list[0], labels[0])
        
        if args.failures:
            analyze_failures(data_list[0], labels[0])
    else:
        # 다중 파일 비교
        compare_results(data_list, labels)
        
        if args.failures:
            for data, label in zip(data_list, labels):
                analyze_failures(data, label)
    
    # 차트 생성
    if args.plot:
        plot_results(data_list, labels)
    
    print(f"\n분석 완료.")


if __name__ == '__main__':
    main()
