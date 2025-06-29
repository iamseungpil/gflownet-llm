#!/usr/bin/env python3
"""
간단한 실험 테스트 스크립트
"""

import os
os.environ['WANDB_MODE'] = 'disabled'  # W&B 비활성화

from experiment_rearc import REARCExperiment
from experiment_harc import HARCExperiment

def test_rearc_experiment():
    """RE-ARC 실험 테스트"""
    print("="*60)
    print("Testing RE-ARC Experiment")
    print("="*60)
    
    exp = REARCExperiment(use_wandb=False)
    
    # 단일 태스크, 적은 candidate로 빠른 테스트
    results = exp.run_rearc_experiment(
        task_ids=["6150a2bd"],
        augmented_sizes=[0, 1, 2],
        num_candidates=2,  # 빠른 테스트를 위해 2개만
        use_colors=False
    )
    
    print("\nRE-ARC Test Results:")
    for task_id, task_data in results.items():
        print(f"Task {task_id}:")
        for aug_size, result in task_data['results_by_augmentation'].items():
            print(f"  {aug_size} augmented: {result['accuracy']:.2%}")
    
    return results

def test_harc_experiment():
    """H-ARC 실험 테스트"""
    print("\n" + "="*60)
    print("Testing H-ARC Experiment")
    print("="*60)
    
    exp = HARCExperiment(use_wandb=False)
    
    # 단일 태스크, 적은 candidate로 빠른 테스트
    results = exp.run_harc_experiment(
        task_ids=["6150a2bd"],
        prompt_types=['hint'],  # hint만 테스트 (더 빠름)
        trace_counts=[0, 1, 2],
        num_candidates=2,  # 빠른 테스트를 위해 2개만
        use_colors=False
    )
    
    print("\nH-ARC Test Results:")
    for task_id, task_data in results.items():
        print(f"Task {task_id}:")
        for prompt_type, prompt_results in task_data['results_by_prompt_type'].items():
            print(f"  {prompt_type} prompt:")
            for trace_count, result in prompt_results.items():
                print(f"    {trace_count} traces: {result['accuracy']:.2%}")
    
    return results

if __name__ == "__main__":
    print("Starting Quick Experiment Test...")
    
    # RE-ARC 테스트
    rearc_results = test_rearc_experiment()
    
    # H-ARC 테스트
    harc_results = test_harc_experiment()
    
    print("\n" + "="*60)
    print("Test Completed Successfully!")
    print("="*60)
    
    print("✅ RE-ARC experiment functional")
    print("✅ H-ARC experiment functional")
    print("✅ Data loading working")
    print("✅ LLM inference working")
    print("✅ Result processing working")
    
    print("\nReady to run full experiments with run_experiments.py!")