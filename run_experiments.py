import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List
import numpy as np

from experiment_rearc import REARCExperiment
from experiment_harc import HARCExperiment
from experiment_harc_action import HARCActionSequenceExperiment

def plot_comparison_results(rearc_results: Dict, harc_grid_results: Dict, 
                          harc_action_results: Dict, output_dir: str = "results"):
    """RE-ARC, H-ARC 그리드 생성, H-ARC action sequence 실험 결과 비교 플롯"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 태스크별 비교
    for task_id in rearc_results.keys():
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 1. RE-ARC 결과 플롯
        if task_id in rearc_results:
            rearc_data = rearc_results[task_id]['results_by_augmentation']
            aug_sizes = sorted(rearc_data.keys())
            rearc_accuracies = [rearc_data[size]['accuracy'] for size in aug_sizes]
            
            axes[0].plot(aug_sizes, rearc_accuracies, 'b-o', linewidth=2, markersize=8)
            axes[0].set_xlabel('Number of RE-ARC Examples')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title(f'RE-ARC Performance on Task {task_id}')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 1.1)
        
        # 2. H-ARC 그리드 생성 결과 플롯
        if task_id in harc_grid_results:
            harc_data = harc_grid_results[task_id]['results_by_prompt_type']
            
            for prompt_type, results in harc_data.items():
                trace_counts = sorted(results.keys())
                harc_accuracies = [results[count]['accuracy'] for count in trace_counts]
                
                label = 'Full Action Trace' if prompt_type == 'full_trace' else 'Action Hint'
                marker = 's' if prompt_type == 'full_trace' else '^'
                axes[1].plot(trace_counts, harc_accuracies, f'-{marker}', 
                           linewidth=2, markersize=8, label=label)
            
            axes[1].set_xlabel('Number of Human Traces')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title(f'H-ARC Grid Generation on Task {task_id}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1.1)
        
        # 3. H-ARC Action Sequence 결과 플롯
        if task_id in harc_action_results:
            action_data = harc_action_results[task_id]['results_by_num_examples']
            num_examples = sorted(action_data.keys())
            similarities = [action_data[n]['avg_similarity'] for n in num_examples]
            exact_matches = [action_data[n]['exact_match_rate'] for n in num_examples]
            
            ax3 = axes[2]
            ax3_2 = ax3.twinx()
            
            line1 = ax3.plot(num_examples, similarities, 'g-o', linewidth=2, 
                            markersize=8, label='Avg Similarity')
            line2 = ax3_2.plot(num_examples, exact_matches, 'r-s', linewidth=2, 
                              markersize=8, label='Exact Match Rate')
            
            ax3.set_xlabel('Number of Example Sequences')
            ax3.set_ylabel('Average Similarity', color='g')
            ax3_2.set_ylabel('Exact Match Rate', color='r')
            ax3.set_title(f'H-ARC Action Prediction on Task {task_id}')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.1)
            ax3_2.set_ylim(0, 1.1)
            
            # 범례 합치기
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_{task_id}_all.png", dpi=150)
        plt.close()
    
    # 전체 요약 플롯
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 각 태스크의 최고 성능 비교
    task_ids = []
    rearc_best = []
    harc_grid_best = []
    harc_action_best = []
    
    for task_id in rearc_results.keys():
        task_ids.append(task_id)
        
        # RE-ARC 최고 성능
        if task_id in rearc_results:
            rearc_accs = [data['accuracy'] for data in rearc_results[task_id]['results_by_augmentation'].values()]
            rearc_best.append(max(rearc_accs) if rearc_accs else 0)
        else:
            rearc_best.append(0)
        
        # H-ARC 그리드 생성 최고 성능
        if task_id in harc_grid_results:
            harc_accs = []
            for prompt_results in harc_grid_results[task_id]['results_by_prompt_type'].values():
                harc_accs.extend([data['accuracy'] for data in prompt_results.values()])
            harc_grid_best.append(max(harc_accs) if harc_accs else 0)
        else:
            harc_grid_best.append(0)
        
        # H-ARC Action 최고 성능 (similarity 기준)
        if task_id in harc_action_results:
            action_sims = [data['avg_similarity'] for data in harc_action_results[task_id]['results_by_num_examples'].values()]
            harc_action_best.append(max(action_sims) if action_sims else 0)
        else:
            harc_action_best.append(0)
    
    x = np.arange(len(task_ids))
    width = 0.25
    
    ax.bar(x - width, rearc_best, width, label='RE-ARC (Grid)', color='skyblue')
    ax.bar(x, harc_grid_best, width, label='H-ARC (Grid)', color='lightcoral')
    ax.bar(x + width, harc_action_best, width, label='H-ARC (Action Seq)', color='lightgreen')
    
    ax.set_xlabel('Task ID')
    ax.set_ylabel('Best Performance')
    ax.set_title('Best Performance Comparison: All Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(task_ids, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_comparison_all.png", dpi=150)
    plt.close()

def create_summary_report(rearc_results: Dict, harc_grid_results: Dict, 
                         harc_action_results: Dict, output_file: str = "experiment_summary_all.md"):
    """실험 결과 요약 리포트 생성 (action sequence 포함)"""
    
    with open(output_file, 'w') as f:
        f.write("# ARC-LLM Augmentation Experiments Summary (with Action Sequence)\n\n")
        
        # 1. RE-ARC 결과
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
        
        # 2. H-ARC 그리드 생성 결과
        f.write("\n## 2. H-ARC Grid Generation Results\n\n")
        f.write("H-ARC uses human action traces for providing solution hints.\n\n")
        
        for task_id, task_data in harc_grid_results.items():
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
        
        # 3. H-ARC Action Sequence 결과
        f.write("\n## 3. H-ARC Action Sequence Prediction Results\n\n")
        f.write("Predicting action sequences instead of direct grid generation.\n\n")
        
        for task_id, task_data in harc_action_results.items():
            f.write(f"### Task {task_id}\n")
            f.write(f"- Number of human traces: {task_data['num_human_traces']}\n")
            f.write(f"- Average trace length: {task_data['avg_trace_length']:.1f}\n")
            f.write(f"- Min/Max trace length: {task_data['min_trace_length']}/{task_data['max_trace_length']}\n\n")
            
            f.write("| Example Sequences | Avg Similarity | Common Actions | Exact Match Rate |\n")
            f.write("|-------------------|----------------|----------------|------------------|\n")
            
            for num_ex in sorted(task_data['results_by_num_examples'].keys()):
                result = task_data['results_by_num_examples'][num_ex]
                f.write(f"| {num_ex} | {result['avg_similarity']:.3f} | {result['avg_common_actions_ratio']:.3f} | {result['exact_match_rate']:.2%} |\n")
            
            f.write("\n")
        
        # 4. 주요 발견사항
        f.write("\n## 4. Key Findings\n\n")
        
        f.write("### Best Performance by Method\n\n")
        f.write("| Task ID | RE-ARC (Grid) | H-ARC (Grid) | H-ARC (Action) | Best Method |\n")
        f.write("|---------|---------------|--------------|----------------|-------------|\n")
        
        for task_id in rearc_results.keys():
            # RE-ARC 최고 성능
            rearc_best = 0
            if task_id in rearc_results:
                rearc_accs = [data['accuracy'] for data in rearc_results[task_id]['results_by_augmentation'].values()]
                rearc_best = max(rearc_accs) if rearc_accs else 0
            
            # H-ARC 그리드 최고 성능
            harc_grid_best = 0
            if task_id in harc_grid_results:
                harc_accs = []
                for prompt_results in harc_grid_results[task_id]['results_by_prompt_type'].values():
                    harc_accs.extend([data['accuracy'] for data in prompt_results.values()])
                harc_grid_best = max(harc_accs) if harc_accs else 0
            
            # H-ARC Action 최고 성능
            harc_action_best = 0
            if task_id in harc_action_results:
                action_sims = [data['avg_similarity'] for data in harc_action_results[task_id]['results_by_num_examples'].values()]
                harc_action_best = max(action_sims) if action_sims else 0
            
            # 최고 성능 방법 결정
            performances = {
                'RE-ARC': rearc_best,
                'H-ARC Grid': harc_grid_best,
                'H-ARC Action': harc_action_best
            }
            best_method = max(performances, key=performances.get)
            
            f.write(f"| {task_id} | {rearc_best:.2%} | {harc_grid_best:.2%} | {harc_action_best:.3f} | {best_method} |\n")
        
        f.write("\n### Observations\n\n")
        f.write("1. **Grid Generation**: Compare RE-ARC vs H-ARC approaches for direct grid generation\n")
        f.write("2. **Action Sequence**: How well can we predict human solving strategies?\n")
        f.write("3. **Data Efficiency**: Which method needs fewer examples?\n")
        f.write("4. **Task Dependency**: Some tasks may benefit more from certain approaches\n")

def run_all_experiments_with_action():
    """모든 실험 실행 (action sequence 포함)"""
    
    print("="*60)
    print("Starting ARC-LLM Augmentation Experiments (with Action Sequence)")
    print("="*60)
    
    # 공통 태스크 설정
    task_ids = ["74dd1130"]
    
    # 1. RE-ARC 실험
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
    
    # 2. H-ARC 그리드 생성 실험
    print("\n\n" + "="*60)
    print("PHASE 2: H-ARC Grid Generation Experiments")
    print("="*60)
    
    harc_experiment = HARCExperiment(use_wandb=True)
    harc_grid_results = harc_experiment.run_harc_experiment(
        task_ids=task_ids,
        prompt_types=['full_trace', 'hint'],
        trace_counts=[0, 1, 2, 3, 5],
        num_candidates=5,
        use_colors=False
    )
    harc_experiment.save_results(harc_grid_results, "harc_grid_results.json")
    
    # 3. H-ARC Action Sequence 실험
    print("\n\n" + "="*60)
    print("PHASE 3: H-ARC Action Sequence Experiments")
    print("="*60)
    
    harc_action_experiment = HARCActionSequenceExperiment(use_wandb=True)
    harc_action_results = harc_action_experiment.run_action_sequence_experiment(
        task_ids=task_ids,
        num_examples=[0, 1, 3],
        num_candidates=5
    )
    harc_action_experiment.save_results(harc_action_results, "harc_action_results.json")
    
    # 4. 결과 분석 및 비교
    print("\n\n" + "="*60)
    print("PHASE 4: Results Analysis and Comparison")
    print("="*60)
    
    # 결과 플롯 생성
    plot_comparison_results(rearc_results, harc_grid_results, harc_action_results)
    print("✓ Comparison plots saved to results/")
    
    # 요약 리포트 생성
    create_summary_report(rearc_results, harc_grid_results, harc_action_results)
    print("✓ Summary report saved to experiment_summary_all.md")
    
    # 최종 요약 출력
    print("\n\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    
    for task_id in task_ids:
        print(f"\nTask {task_id}:")
        
        # RE-ARC 최고 성능
        if task_id in rearc_results:
            rearc_accs = [data['accuracy'] for data in rearc_results[task_id]['results_by_augmentation'].values()]
            rearc_best = max(rearc_accs) if rearc_accs else 0
            print(f"  RE-ARC best accuracy: {rearc_best:.2%}")
        
        # H-ARC 그리드 최고 성능
        if task_id in harc_grid_results:
            harc_accs = []
            for prompt_results in harc_grid_results[task_id]['results_by_prompt_type'].values():
                harc_accs.extend([data['accuracy'] for data in prompt_results.values()])
            harc_best = max(harc_accs) if harc_accs else 0
            print(f"  H-ARC Grid best accuracy: {harc_best:.2%}")
        
        # H-ARC Action 최고 성능
        if task_id in harc_action_results:
            action_sims = [data['avg_similarity'] for data in harc_action_results[task_id]['results_by_num_examples'].values()]
            action_best = max(action_sims) if action_sims else 0
            print(f"  H-ARC Action best similarity: {action_best:.3f}")
    
    print("\n✅ All experiments completed successfully!")
    
    # Wandb 종료
    if rearc_experiment.use_wandb:
        import wandb
        wandb.finish()
    if harc_experiment.use_wandb:
        import wandb
        wandb.finish()
    if harc_action_experiment.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    run_all_experiments_with_action()
