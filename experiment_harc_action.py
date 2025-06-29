"""
H-ARC Action Sequence Prediction Extension
기존 그리드 생성 실험에 추가로 action sequence 예측 실험을 수행합니다.
"""

import re
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import numpy as np

# 기존 experiment_harc.py를 확장
from experiment_harc import HARCExperiment


class HARCActionSequenceExperiment(HARCExperiment):
    """H-ARC Action Sequence 예측 실험을 위한 확장 클래스"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Action 매핑 정의
        self.action_mapping = {
            'click_cell': ['click', 'select_cell', 'cell_click'],
            'select_color': ['color', 'pick_color', 'choose_color'],
            'flood_fill': ['fill', 'bucket_fill', 'paint'],
            'copy': ['copy', 'copy_selection'],
            'paste': ['paste', 'paste_selection'],
            'resize_grid': ['resize', 'resize_output', 'change_size'],
            'rotate': ['rotate', 'rotation', 'turn'],
            'flip': ['flip', 'mirror'],
            'submit': ['submit', 'done', 'finish']
        }
    
    def create_action_sequence_prompt(self, task, action_traces: List[List[Dict]], 
                                     num_examples: int = 3) -> str:
        """Action sequence 예측을 위한 프롬프트 생성"""
        prompt = """You are analyzing how humans solve ARC (Abstraction and Reasoning Corpus) tasks.
Given the training examples, predict the sequence of actions needed to solve the test case.

Available actions:
- click_cell: Click on a specific cell
- select_color: Select a color (0-9)
- flood_fill: Fill an area with the selected color
- copy: Copy a selected region
- paste: Paste the copied region
- resize_grid: Resize the output grid
- rotate: Rotate a selection
- flip: Flip a selection
- submit: Submit the solution

Training Examples:
"""
        
        # 원본 예제들
        for i, pair in enumerate(task.train_pairs):
            prompt += f"\nExample {i+1}:\nInput:\n{self.grid_to_string(pair.x.tolist())}\n"
            prompt += f"Output:\n{self.grid_to_string(pair.y.tolist())}\n"
        
        # 인간의 실제 action sequence 예시
        if action_traces and num_examples > 0:
            prompt += "\n\nHow humans solved this task (example action sequences):"
            
            for i, trace in enumerate(action_traces[:num_examples]):
                prompt += f"\n\nHuman Solution {i+1}:"
                
                # 전체 action sequence를 보여줌
                for j, action in enumerate(trace):
                    action_type = action['action']
                    prompt += f"\n{j+1}. {action_type}"
                    
                    # 추가 정보가 있으면 포함
                    if action.get('selected_symbol') is not None:
                        prompt += f" (color: {action['selected_symbol']})"
                    if action.get('selected_tool'):
                        prompt += f" (tool: {action['selected_tool']})"
        
        # 테스트 케이스
        prompt += f"\n\nNow predict the action sequence for this test case:"
        prompt += f"\nTest Input:\n{self.grid_to_string(task.test_pairs[0].x.tolist())}\n"
        prompt += f"Expected Output:\n{self.grid_to_string(task.test_pairs[0].y.tolist())}\n"
        
        prompt += "\nList the actions needed to transform the input to the output."
        prompt += "\nFormat each action on a new line with a number, like:\n1. action_name\n2. action_name\n..."
        
        return prompt
    
    def parse_predicted_actions(self, response: str) -> List[str]:
        """LLM 응답에서 action sequence 파싱"""
        lines = response.strip().split('\n')
        actions = []
        
        for line in lines:
            line = line.strip()
            # 번호가 있는 라인 찾기 (1. action, 2. action 등)
            if re.match(r'^\d+\.', line):
                # 번호 제거하고 action 추출
                action = re.sub(r'^\d+\.\s*', '', line)
                # 괄호 안의 추가 정보 제거
                action = re.sub(r'\s*\([^)]*\)', '', action)
                action = action.strip().lower().replace(' ', '_')
                
                # action 정규화
                normalized_action = self.normalize_action(action)
                if normalized_action:
                    actions.append(normalized_action)
        
        return actions
    
    def normalize_action(self, action: str) -> str:
        """Action 이름을 정규화"""
        action_lower = action.lower().strip()
        
        # 직접 매칭
        for standard_action, variants in self.action_mapping.items():
            if action_lower == standard_action:
                return standard_action
            for variant in variants:
                if variant in action_lower or action_lower in variant:
                    return standard_action
        
        # 매칭되지 않으면 원본 반환
        return action
    
    def evaluate_action_sequence(self, predicted: List[str], actual_traces: List[List[Dict]]) -> Dict:
        """예측된 action sequence를 실제 인간의 sequence와 비교"""
        if not predicted or not actual_traces:
            return {
                'exact_match': 0, 
                'similarity': 0, 
                'common_actions_ratio': 0,
                'predicted_length': len(predicted) if predicted else 0
            }
        
        # 실제 action sequence들을 문자열 리스트로 변환
        actual_sequences = []
        for trace in actual_traces:
            sequence = [self.normalize_action(action['action']) for action in trace]
            actual_sequences.append(sequence)
        
        # 각 실제 sequence와 비교
        max_similarity = 0
        best_match_info = {}
        
        for actual_seq in actual_sequences:
            # 1. Exact match 체크
            if predicted == actual_seq:
                return {
                    'exact_match': 1.0,
                    'similarity': 1.0,
                    'common_actions_ratio': 1.0,
                    'predicted_length': len(predicted),
                    'actual_length': len(actual_seq),
                    'match_type': 'exact'
                }
            
            # 2. Sequence similarity (SequenceMatcher)
            matcher = SequenceMatcher(None, predicted, actual_seq)
            similarity = matcher.ratio()
            
            # 3. Common actions
            common = set(predicted) & set(actual_seq)
            common_ratio = len(common) / max(len(set(predicted)), len(set(actual_seq))) if max(len(set(predicted)), len(set(actual_seq))) > 0 else 0
            
            # 4. Order-aware similarity (LCS - Longest Common Subsequence)
            lcs_length = self.longest_common_subsequence(predicted, actual_seq)
            lcs_ratio = lcs_length / max(len(predicted), len(actual_seq)) if max(len(predicted), len(actual_seq)) > 0 else 0
            
            # 종합 점수
            combined_similarity = (similarity + common_ratio + lcs_ratio) / 3
            
            if combined_similarity > max_similarity:
                max_similarity = combined_similarity
                best_match_info = {
                    'exact_match': 0.0,
                    'similarity': similarity,
                    'common_actions_ratio': common_ratio,
                    'lcs_ratio': lcs_ratio,
                    'combined_similarity': combined_similarity,
                    'predicted_length': len(predicted),
                    'actual_length': len(actual_seq),
                    'common_actions': list(common),
                    'match_type': 'partial'
                }
        
        return best_match_info
    
    def longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """두 시퀀스의 LCS 길이 계산"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def run_action_sequence_experiment(self, task_ids: List[str] = None,
                                     num_examples: List[int] = [0, 1, 3],
                                     num_candidates: int = 5) -> Dict:
        """Action sequence 예측 실험"""
        if task_ids is None:
            task_ids = ["6150a2bd", "87a80de6", "9ddd00f0", "d43fd935"]
        
        all_results = {}
        
        for task_id in task_ids:
            print(f"\n{'='*60}")
            print(f"Running H-ARC Action Sequence Prediction on task: {task_id}")
            print(f"{'='*60}")
            
            # 태스크와 action trace 로드
            task = self.load_arc_task(task_id)
            if not task:
                print(f"Task {task_id} not found!")
                continue
            
            action_traces = self.load_harc_action_traces(task_id)
            
            if not action_traces:
                print(f"No action traces found for task {task_id}")
                continue
            
            # Action trace 통계
            trace_lengths = [len(trace) for trace in action_traces]
            
            task_results = {
                'task_id': task_id,
                'num_human_traces': len(action_traces),
                'avg_trace_length': sum(trace_lengths) / len(trace_lengths),
                'min_trace_length': min(trace_lengths),
                'max_trace_length': max(trace_lengths),
                'results_by_num_examples': {}
            }
            
            # 다양한 예시 개수로 실험
            for num_ex in num_examples:
                if num_ex > len(action_traces):
                    continue
                
                print(f"\nTesting with {num_ex} example action sequences...")
                
                evaluation_results = []
                
                for candidate_idx in range(num_candidates):
                    # 프롬프트 생성
                    prompt = self.create_action_sequence_prompt(task, action_traces, num_ex)
                    
                    # LLM으로 action sequence 예측
                    response = self.generate_response(prompt, max_tokens=512, temperature=0.3)
                    
                    # 예측된 action 파싱
                    predicted_actions = self.parse_predicted_actions(response)
                    
                    # 평가
                    eval_result = self.evaluate_action_sequence(predicted_actions, action_traces)
                    eval_result['candidate_idx'] = candidate_idx
                    eval_result['predicted_actions'] = predicted_actions
                    evaluation_results.append(eval_result)
                
                # 평균 점수 계산
                avg_similarity = sum(r.get('combined_similarity', r.get('similarity', 0)) 
                                   for r in evaluation_results) / len(evaluation_results)
                exact_matches = sum(r.get('exact_match', 0) for r in evaluation_results)
                avg_common_ratio = sum(r.get('common_actions_ratio', 0) 
                                     for r in evaluation_results) / len(evaluation_results)
                
                task_results['results_by_num_examples'][num_ex] = {
                    'avg_similarity': avg_similarity,
                    'avg_common_actions_ratio': avg_common_ratio,
                    'exact_match_rate': exact_matches / num_candidates,
                    'num_candidates': num_candidates,
                    'detailed_results': evaluation_results
                }
                
                print(f"Average similarity: {avg_similarity:.3f}")
                print(f"Average common actions ratio: {avg_common_ratio:.3f}")
                print(f"Exact match rate: {exact_matches}/{num_candidates}")
                
                # 예시 출력
                if evaluation_results:
                    best_result = max(evaluation_results, 
                                    key=lambda x: x.get('combined_similarity', x.get('similarity', 0)))
                    print(f"\nBest predicted sequence (length: {best_result.get('predicted_length', 0)}):")
                    for i, action in enumerate(best_result.get('predicted_actions', [])[:10]):
                        print(f"  {i+1}. {action}")
                    if len(best_result.get('predicted_actions', [])) > 10:
                        print(f"  ... ({len(best_result.get('predicted_actions', [])) - 10} more actions)")
                
                if self.use_wandb:
                    wandb.log({
                        'task_id': task_id,
                        'num_examples': num_ex,
                        'avg_similarity': avg_similarity,
                        'avg_common_actions_ratio': avg_common_ratio,
                        'exact_match_rate': exact_matches / num_candidates,
                        'experiment': 'h-arc_action_sequence'
                    })
            
            all_results[task_id] = task_results
        
        return all_results
    
    def run_combined_experiment(self, task_ids: List[str] = None) -> Dict:
        """그리드 생성과 action sequence 예측 실험을 모두 실행"""
        if task_ids is None:
            task_ids = ["6150a2bd", "178fcbfb", "1190e5a7", "150deff5"]
        
        combined_results = {
            'grid_generation': {},
            'action_sequence': {}
        }
        
        print("\n" + "="*60)
        print("Running H-ARC Grid Generation Experiments")
        print("="*60)
        
        # 그리드 생성 실험
        grid_results = self.run_harc_experiment(
            task_ids=task_ids,
            prompt_types=['full_trace', 'hint'],
            trace_counts=[0, 1, 3, 5],
            num_candidates=5,
            use_colors=False
        )
        combined_results['grid_generation'] = grid_results
        
        print("\n" + "="*60)
        print("Running H-ARC Action Sequence Prediction Experiments")
        print("="*60)
        
        # Action sequence 예측 실험
        action_results = self.run_action_sequence_experiment(
            task_ids=task_ids,
            num_examples=[0, 1, 3],
            num_candidates=5
        )
        combined_results['action_sequence'] = action_results
        
        # 결과 비교 출력
        print("\n" + "="*60)
        print("EXPERIMENT COMPARISON")
        print("="*60)
        
        for task_id in task_ids:
            print(f"\nTask: {task_id}")
            
            # 그리드 생성 결과
            if task_id in grid_results:
                grid_task = grid_results[task_id]
                print("\n  Grid Generation Results:")
                for prompt_type in ['full_trace', 'hint']:
                    if prompt_type in grid_task.get('results_by_prompt_type', {}):
                        results = grid_task['results_by_prompt_type'][prompt_type]
                        accuracies = [r['accuracy'] for r in results.values()]
                        if accuracies:
                            best_acc = max(accuracies)
                            print(f"    {prompt_type}: best accuracy = {best_acc:.2%}")
            
            # Action sequence 결과
            if task_id in action_results:
                action_task = action_results[task_id]
                print("\n  Action Sequence Prediction Results:")
                print(f"    Human traces: {action_task['num_human_traces']}")
                print(f"    Avg trace length: {action_task['avg_trace_length']:.1f}")
                
                results = action_task.get('results_by_num_examples', {})
                if results:
                    best_sim = max(r['avg_similarity'] for r in results.values())
                    print(f"    Best similarity score: {best_sim:.3f}")
                    for num_ex, result in sorted(results.items()):
                        print(f"    {num_ex} examples: similarity = {result['avg_similarity']:.3f}, "
                              f"exact match = {result['exact_match_rate']:.2%}")
        
        return combined_results


def main():
    """메인 실행 함수"""
    import wandb
    
    # 실험 초기화
    experiment = HARCActionSequenceExperiment(use_wandb=True)
    
    # 실험할 태스크들
    task_ids = ["6150a2bd", "178fcbfb", "1190e5a7", "150deff5"]
    
    # 전체 실험 실행 (그리드 생성 + action sequence)
    combined_results = experiment.run_combined_experiment(task_ids)
    
    # 결과 저장
    experiment.save_results(combined_results['grid_generation'], "harc_grid_results.json")
    experiment.save_results(combined_results['action_sequence'], "harc_action_results.json")
    experiment.save_results(combined_results, "harc_combined_results.json")
    
    if experiment.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
