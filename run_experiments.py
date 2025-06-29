import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List
import numpy as np

from experiment_rearc import REARCExperiment
from experiment_harc import HARCExperiment

def plot_comparison_results(rearc_results: Dict, harc_results: Dict, output_dir: str = "results"):
    """RE-ARC와 H-ARC 실험 결과 비교 플롯"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 태스크별 정확도 비교
    for task_id in rearc_results.keys():
        if task_id not in harc_results:
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RE-ARC 결과 플롯
        rearc_data = rearc_results[task_id]['results_by_augmentation']
        aug_sizes = sorted(rearc_data.keys())
        rearc_accuracies = [rearc_data[size]['accuracy'] for size in aug_sizes]
        
        ax1.plot(aug_sizes, rearc_accuracies, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of RE-ARC Augmented Examples')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'RE-ARC Performance on Task {task_id}')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # H-ARC 결과 플롯
        harc_data = harc_results[task_id]['results_by_prompt_type']
        
        for prompt_type, results in harc_data.items():
            trace_counts = sorted(results.keys())
            harc_accuracies = [results[count]['accuracy'] for count in trace_counts]
            
            label = 'Full Action Trace' if prompt_type == 'full_trace' else 'Action Hint'
            marker = 's' if prompt_type == 'full_trace' else '^'
            ax2.plot(trace_counts, harc_accuracies, f'-{marker}', linewidth=2, markersize=8, label=label)
        
        ax2.set_xlabel('Number of Human Traces')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'H-ARC Performance on Task {task_id}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_{task_id}.png", dpi=150)
        plt.close()
    
    # 전체 요약 플롯
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 각 태스크의 최고 성능 비교
    task_ids = []
    rearc_best = []
    harc_best = []
    
    for task_id in rearc_results.keys():
        if task_id not in harc_results:
            continue
        
        task_ids.append(task_id)
        
        # RE-ARC 최고 성능
        rearc_accs = [data['accuracy'] for data in rearc_results[task_id]['results_by_augmentation'].values()]
        rearc_best.append(max(rearc_accs) if rearc_accs else 0)
        
        # H-ARC 최고 성능
        harc_accs = []
        for prompt_results in harc_results[task_id]['results_by_prompt_type'].values():
            harc_accs.extend([data['accuracy'] for data in prompt_results.values()])
        harc_best.append(max(harc_accs) if harc_accs else 0)
    
    x = np.arange(len(task_ids))
    width = 0.35
    
    ax.bar(x - width/2, rearc_best, width, label='RE-ARC (Best)', color='skyblue')
    ax.bar(x + width/2, harc_best, width, label='H-ARC (Best)', color='lightcoral')
    
    ax.set_xlabel('Task ID')
    ax.set_ylabel('Best Accuracy')
    ax.set_title('Best Performance Comparison: RE-ARC vs H-ARC')
    ax.set_xticks(x)
    ax.set_xticklabels(task_ids, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_comparison.png", dpi=150)
    plt.close()

def create_summary_report(rearc_results: Dict, harc_results: Dict, output_file: str = "experiment_summary.md"):
    """실험 결과 요약 리포트 생성"""
    
    with open(output_file, 'w') as f:
        f.write("# ARC-LLM Augmentation Experiments Summary\n\n")
        
        f.write("## 1. RE-ARC Experiment Results\n\n")
        f.write("RE-ARC uses procedurally generated input-output grid pairs for augmentation.\n\n")
        
        for task_id, task_data in rearc_results.items():
            f.write(f"### Task {task_id}\n")
            f.write(f"- Number of available RE-ARC examples: {task_data['num_rearc_examples']}\n")
            f.write("- Results by augmentation size:\n\n")
            
            f.write("| Augmented Examples | Accuracy | Correct/Total |\n")
            f.write("|-------------------|----------|---------------|\n")
            
            for aug_size in sorted(task_data['results_by_augmentation'].keys()):
                result = task_data['results_by_augmentation'][aug_size]
                f.write(f"| {aug_size} | {result['accuracy']:.2%} | {result['correct_count']}/{result['total_candidates']} |\n")
            
            f.write("\n")
        
        f.write("\n## 2. H-ARC Experiment Results\n\n")
        f.write("H-ARC uses human action traces for providing solution hints.\n\n")
        
        for task_id, task_data in harc_results.items():
            f.write(f"### Task {task_id}\n")
            f.write(f"- Number of available human traces: {task_data['num_human_traces']}\n\n")
            
            for prompt_type, results in task_data['results_by_prompt_type'].items():
                f.write(f"#### {prompt_type.replace('_', ' ').title()} Prompt\n\n")
                
                f.write("| Human Traces | Accuracy | Correct/Total |\n")
                f.write("|--------------|----------|---------------|\n")
                
                for num_traces in sorted(results.keys()):
                    result = results[num_traces]
                    f.write(f"| {num_traces} | {result['accuracy']:.2%} | {result['correct_count']}/{result['total_candidates']} |\n")
                
                f.write("\n")
        
        f.write("\n## 3. Key Findings\n\n")
        
        # 최고 성능 분석
        f.write("### Best Performance by Method\n\n")
        f.write("| Task ID | RE-ARC Best | H-ARC Best | Better Method |\n")
        f.write("|---------|-------------|------------|---------------|\n")
        
        for task_id in rearc_results.keys():
            if task_id not in harc_results:
                continue
            
            # RE-ARC 최고 성능
            rearc_accs = [data['accuracy'] for data in rearc_results[task_id]['results_by_augmentation'].values()]
            rearc_best = max(rearc_accs) if rearc_accs else 0
            
            # H-ARC 최고 성능
            harc_accs = []
            for prompt_results in harc_results[task_id]['results_by_prompt_type'].values():
                harc_accs.extend([data['accuracy'] for data in prompt_results.values()])
            harc_best = max(harc_accs) if harc_accs else 0
            
            better = "RE-ARC" if rearc_best > harc_best else ("H-ARC" if harc_best > rearc_best else "Tie")
            f.write(f"| {task_id} | {rearc_best:.2%} | {harc_best:.2%} | {better} |\n")
        
        f.write("\n### Observations\n\n")
        f.write("1. **RE-ARC Performance**: Grid-based augmentation shows [describe pattern]\n")
        f.write("2. **H-ARC Performance**: Action trace guidance shows [describe pattern]\n")
        f.write("3. **Data Efficiency**: [Compare how many examples needed for good performance]\n")
        f.write("4. **Prompt Strategy**: [Compare full trace vs hint effectiveness]\n")

def run_all_experiments():
    """모든 실험 실행 및 결과 비교"""
    
    print("="*60)
    print("Starting ARC-LLM Augmentation Experiments")
    print("="*60)
    
    # 공통 태스크 설정
    task_ids = ["6150a2bd", "178fcbfb", "1190e5a7", "150deff5"]
    
    # RE-ARC 실험
    print("\n\n" + "="*60)
    print("PHASE 1: RE-ARC Experiments")
    print("="*60)
    
    rearc_experiment = REARCExperiment(use_wandb=True)
    rearc_results = rearc_experiment.run_rearc_experiment(
        task_ids=task_ids,
        augmented_sizes=[0, 1, 2, 3, 5, 10],
        num_candidates=5,
        use_colors=False
    )
    rearc_experiment.save_results(rearc_results, "rearc_results.json")
    
    # H-ARC 실험
    print("\n\n" + "="*60)
    print("PHASE 2: H-ARC Experiments")
    print("="*60)
    
    harc_experiment = HARCExperiment(use_wandb=True)
    harc_results = harc_experiment.run_harc_experiment(
        task_ids=task_ids,
        prompt_types=['full_trace', 'hint'],
        trace_counts=[0, 1, 2, 3, 5],
        num_candidates=5,
        use_colors=False
    )
    harc_experiment.save_results(harc_results, "harc_results.json")
    
    # 결과 비교 및 시각화
    print("\n\n" + "="*60)
    print("PHASE 3: Results Analysis and Comparison")
    print("="*60)
    
    # 결과 플롯 생성
    plot_comparison_results(rearc_results, harc_results)
    print("✓ Comparison plots saved to results/")
    
    # 요약 리포트 생성
    create_summary_report(rearc_results, harc_results)
    print("✓ Summary report saved to experiment_summary.md")
    
    # 최종 요약 출력
    print("\n\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    
    for task_id in task_ids:
        if task_id in rearc_results and task_id in harc_results:
            print(f"\nTask {task_id}:")
            
            # RE-ARC 최고 성능
            rearc_accs = [data['accuracy'] for data in rearc_results[task_id]['results_by_augmentation'].values()]
            rearc_best = max(rearc_accs) if rearc_accs else 0
            print(f"  RE-ARC best accuracy: {rearc_best:.2%}")
            
            # H-ARC 최고 성능
            harc_accs = []
            for prompt_results in harc_results[task_id]['results_by_prompt_type'].values():
                harc_accs.extend([data['accuracy'] for data in prompt_results.values()])
            harc_best = max(harc_accs) if harc_accs else 0
            print(f"  H-ARC best accuracy: {harc_best:.2%}")
    
    print("\n✅ All experiments completed successfully!")
    
    # Wandb 종료
    if rearc_experiment.use_wandb:
        import wandb
        wandb.finish()
    if harc_experiment.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    run_all_experiments()
