#!/usr/bin/env python3
"""
H-ARC Action Sequence 실험만 실행하는 스크립트
"""

import argparse
from experiment_harc_action import HARCActionSequenceExperiment
import wandb


def main():
    parser = argparse.ArgumentParser(description='Run H-ARC Action Sequence Experiments')
    parser.add_argument('--tasks', nargs='+', 
                       default=["6150a2bd", "178fcbfb", "1190e5a7", "150deff5"],
                       help='Task IDs to experiment on')
    parser.add_argument('--num-examples', nargs='+', type=int,
                       default=[0, 1, 3, 5],
                       help='Number of example sequences to show')
    parser.add_argument('--num-candidates', type=int, default=5,
                       help='Number of candidates per configuration')
    parser.add_argument('--combined', action='store_true',
                       help='Run both grid generation and action sequence experiments')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    print("="*60)
    print("H-ARC Action Sequence Experiments")
    print("="*60)
    print(f"Tasks: {args.tasks}")
    print(f"Example counts: {args.num_examples}")
    print(f"Candidates per config: {args.num_candidates}")
    print(f"Combined experiment: {args.combined}")
    print("="*60)
    
    # 실험 초기화
    experiment = HARCActionSequenceExperiment(use_wandb=not args.no_wandb)
    
    if args.combined:
        # 그리드 생성 + action sequence 실험
        print("\nRunning combined experiments (Grid + Action Sequence)...")
        results = experiment.run_combined_experiment(task_ids=args.tasks)
        
        # 결과 저장
        experiment.save_results(results['grid_generation'], "harc_grid_results.json")
        experiment.save_results(results['action_sequence'], "harc_action_results.json")
        experiment.save_results(results, "harc_combined_results.json")
    else:
        # Action sequence 실험만
        print("\nRunning action sequence prediction experiments only...")
        results = experiment.run_action_sequence_experiment(
            task_ids=args.tasks,
            num_examples=args.num_examples,
            num_candidates=args.num_candidates
        )
        
        # 결과 저장
        experiment.save_results(results, "harc_action_sequence_results.json")
    
    # 결과 요약
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if args.combined and 'action_sequence' in results:
        action_results = results['action_sequence']
    else:
        action_results = results
    
    for task_id in args.tasks:
        if task_id in action_results:
            task_data = action_results[task_id]
            print(f"\nTask: {task_id}")
            print(f"  Human traces available: {task_data['num_human_traces']}")
            print(f"  Avg trace length: {task_data['avg_trace_length']:.1f}")
            
            best_sim = 0
            best_config = None
            for num_ex, result in task_data['results_by_num_examples'].items():
                if result['avg_similarity'] > best_sim:
                    best_sim = result['avg_similarity']
                    best_config = num_ex
            
            print(f"  Best similarity: {best_sim:.3f} (with {best_config} examples)")
            
            # 각 설정별 결과
            for num_ex in sorted(task_data['results_by_num_examples'].keys()):
                result = task_data['results_by_num_examples'][num_ex]
                print(f"    {num_ex} examples: similarity={result['avg_similarity']:.3f}, "
                      f"exact_match={result['exact_match_rate']:.2%}")
    
    print("\n✅ Experiments completed!")
    
    if experiment.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
