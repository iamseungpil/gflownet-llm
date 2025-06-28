import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results_file: str = "scaling_experiment_results.json"):
    """실험 결과를 시각화"""
    
    # 결과 파일 로드
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 정확도 곡선 플롯
    ax1.set_title(f"Accuracy vs. Number of Examples\nTask: {results['task_id']}", fontsize=14)
    ax1.set_xlabel("Number of Examples", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Baseline 플롯
    if 'baseline' in results:
        baseline_data = results['baseline']['accuracy_by_data_size']
        x_baseline = list(map(int, baseline_data.keys()))
        y_baseline = [data['accuracy'] for data in baseline_data.values()]
        ax1.plot(x_baseline, y_baseline, 'o-', label='Baseline', linewidth=2, markersize=8)
    
    # RE-ARC 플롯
    if 're-arc' in results and results['re-arc']['accuracy_by_data_size']:
        rearc_data = results['re-arc']['accuracy_by_data_size']
        x_rearc = [int(k) + len(results['baseline']['accuracy_by_data_size']) 
                   for k in rearc_data.keys()]
        y_rearc = [data['accuracy'] for data in rearc_data.values()]
        ax1.plot(x_rearc, y_rearc, 's-', label='RE-ARC', linewidth=2, markersize=8)
    
    # H-ARC 플롯
    if 'h-arc' in results and results['h-arc']['accuracy_by_data_size']:
        harc_data = results['h-arc']['accuracy_by_data_size']
        x_harc = [int(k) + len(results['baseline']['accuracy_by_data_size']) 
                  for k in harc_data.keys()]
        y_harc = [data['accuracy'] for data in harc_data.values()]
        ax1.plot(x_harc, y_harc, '^-', label='H-ARC', linewidth=2, markersize=8)
    
    ax1.legend(fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    
    # 2. 정확도 변화량 (Delta) 플롯
    ax2.set_title("Accuracy Change (Δ) per Additional Example", fontsize=14)
    ax2.set_xlabel("Example Index", fontsize=12)
    ax2.set_ylabel("Accuracy Delta", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Baseline deltas
    if 'accuracy_deltas' in results.get('baseline', {}):
        deltas = results['baseline']['accuracy_deltas']
        x_delta = range(1, len(deltas) + 1)
        ax2.bar(x_delta, deltas, alpha=0.7, label='Baseline', width=0.25)
    
    # RE-ARC deltas
    if 'accuracy_deltas' in results.get('re-arc', {}):
        deltas = results['re-arc']['accuracy_deltas']
        x_delta = np.arange(1, len(deltas) + 1) + 0.3
        ax2.bar(x_delta, deltas, alpha=0.7, label='RE-ARC', width=0.25)
    
    # H-ARC deltas
    if 'accuracy_deltas' in results.get('h-arc', {}):
        deltas = results['h-arc']['accuracy_deltas']
        x_delta = np.arange(1, len(deltas) + 1) + 0.6
        ax2.bar(x_delta, deltas, alpha=0.7, label='H-ARC', width=0.25)
    
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    
    # 결과 저장
    output_file = results_file.replace('.json', '_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()
    
    # 통계 요약 출력
    print("\n=== Summary Statistics ===")
    print(f"Task ID: {results['task_id']}")
    print(f"Candidates per configuration: {results['num_candidates']}")
    
    # 최고 정확도 찾기
    best_accuracy = 0
    best_method = ""
    
    for method in ['baseline', 're-arc', 'h-arc']:
        if method in results and 'accuracy_by_data_size' in results[method]:
            for config, data in results[method]['accuracy_by_data_size'].items():
                if data['accuracy'] > best_accuracy:
                    best_accuracy = data['accuracy']
                    best_method = f"{method} ({config} examples)"
    
    print(f"\nBest accuracy: {best_accuracy:.2%} - {best_method}")
    
    # 개선률 계산
    if 'baseline' in results and 're-arc' in results:
        baseline_max = max([d['accuracy'] for d in results['baseline']['accuracy_by_data_size'].values()])
        rearc_max = max([d['accuracy'] for d in results['re-arc']['accuracy_by_data_size'].values()])
        improvement = (rearc_max - baseline_max) / baseline_max * 100
        print(f"RE-ARC improvement over baseline: {improvement:+.1f}%")
    
    if 'baseline' in results and 'h-arc' in results:
        baseline_max = max([d['accuracy'] for d in results['baseline']['accuracy_by_data_size'].values()])
        harc_max = max([d['accuracy'] for d in results['h-arc']['accuracy_by_data_size'].values()])
        improvement = (harc_max - baseline_max) / baseline_max * 100
        print(f"H-ARC improvement over baseline: {improvement:+.1f}%")


if __name__ == "__main__":
    plot_results()
