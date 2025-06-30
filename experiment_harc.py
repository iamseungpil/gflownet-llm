"""
Complete H-ARC Experiment Implementation
그리드 생성과 action sequence 예측 실험을 모두 포함한 완전한 H-ARC 실험 클래스
"""

import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re
import wandb
from tqdm import tqdm
from arc import train_problems, validation_problems
from collections import defaultdict
from difflib import SequenceMatcher

# Color mapping for ARC grids
COLOR_MAP = {
    0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
    5: "Gray", 6: "Pink", 7: "Orange", 8: "Purple", 9: "Brown"
}

REVERSE_COLOR_MAP = {v: k for k, v in COLOR_MAP.items()}

class HARCExperiment:
    """
    Complete H-ARC 실험: 그리드 생성과 action sequence 예측을 모두 포함
    """
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", use_wandb=True):
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # H-ARC 데이터 경로
        self.harc_data_path = Path("data/h-arc/data/data.csv")
        self.harc_summary_path = Path("data/h-arc/data/summary_data.csv")
        
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
        
        # W&B 초기화
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
            wandb.init(
                project="arc-llm-harc-experiment",
                config={
                    "model": model_name,
                    "device": str(self.device),
                    "experiment_type": "h-arc_complete"
                }
            )
    
    def load_arc_task(self, task_id: str):
        """원본 ARC 태스크 로드"""
        all_problems = train_problems + validation_problems
        
        for problem in all_problems:
            if problem.uid == task_id:
                return problem
        return None
    
    def load_harc_action_traces(self, task_id: str) -> List[List[Dict]]:
        """H-ARC에서 성공한 참가자들의 action trace 로드"""
        if not self.harc_data_path.exists():
            print(f"H-ARC data file not found: {self.harc_data_path}")
            return []
        
        # 데이터 로드
        df = pd.read_csv(self.harc_data_path)
        
        # 해당 태스크의 데이터 필터링
        task_data = df[df['task_name'] == task_id]
        
        if len(task_data) == 0:
            print(f"No H-ARC data found for task {task_id}")
            return []
        
        # 성공한 시도만 필터링
        successful_attempts = task_data[task_data['solved'] == True]
        
        # 참가자별로 action trace 수집
        action_traces = []
        unique_attempts = successful_attempts[['hashed_id', 'attempt_number']].drop_duplicates()
        
        for _, row in unique_attempts.iterrows():
            participant_id = row['hashed_id']
            attempt_num = row['attempt_number']
            
            # 해당 시도의 모든 action 가져오기
            attempt_data = task_data[
                (task_data['hashed_id'] == participant_id) & 
                (task_data['attempt_number'] == attempt_num)
            ].sort_values('action_id')
            
            # action sequence 추출
            actions = []
            for _, action_row in attempt_data.iterrows():
                action_info = {
                    'action': action_row['action'],
                    'action_id': action_row['action_id'],
                    'selected_tool': action_row.get('selected_tool', None),
                    'selected_symbol': action_row.get('selected_symbol', None),
                    'solved': action_row['solved']
                }
                actions.append(action_info)
            
            if actions:  # 비어있지 않은 경우만 추가
                action_traces.append(actions)
        
        print(f"Loaded {len(action_traces)} successful action traces for task {task_id}")
        return action_traces
    
    def simplify_action_trace(self, actions: List[Dict]) -> List[str]:
        """Action trace를 간단한 설명으로 변환"""
        simplified = []
        
        for action in actions:
            action_type = action['action']
            
            # 주요 action 타입별 설명 생성
            if action_type == 'click_cell':
                desc = "Click on a cell"
            elif action_type == 'select_color':
                color = action.get('selected_symbol', 'unknown')
                desc = f"Select color {color}"
            elif action_type == 'flood_fill':
                desc = "Apply flood fill"
            elif action_type == 'copy':
                desc = "Copy selected area"
            elif action_type == 'paste':
                desc = "Paste copied area"
            elif action_type == 'submit':
                desc = "Submit solution"
            elif action_type == 'reset':
                desc = "Reset grid"
            elif 'resize' in action_type:
                desc = f"Resize grid ({action_type})"
            elif 'rotate' in action_type:
                desc = f"Rotate selection ({action_type})"
            elif 'flip' in action_type:
                desc = f"Flip selection ({action_type})"
            else:
                desc = action_type.replace('_', ' ').capitalize()
            
            simplified.append(desc)
        
        return simplified
    
    def grid_to_string(self, grid: List[List[int]], use_colors: bool = False) -> str:
        """그리드를 문자열로 변환"""
        if use_colors:
            rows = []
            for row in grid:
                color_row = [COLOR_MAP.get(int(cell), f"Unknown_{cell}") for cell in row]
                rows.append(" ".join(color_row))
            return "\n".join(rows)
        else:
            return '\n'.join([' '.join(map(str, row)) for row in grid])
    
    def string_to_grid(self, grid_str: str, from_colors: bool = False) -> List[List[int]]:
        """문자열을 그리드로 변환"""
        try:
            lines = grid_str.strip().split('\n')
            grid = []
            for line in lines:
                if line.strip():
                    if from_colors:
                        colors = line.split()
                        row = [REVERSE_COLOR_MAP.get(color, -1) for color in colors]
                    else:
                        row = [int(x) for x in line.split()]
                    grid.append(row)
            return grid
        except:
            return []
    
    def create_harc_prompt(self, task: Dict, action_traces: List[List[Dict]], 
                          num_traces: int = 3, use_colors: bool = False) -> str:
        """H-ARC action trace를 활용한 프롬프트 생성"""
        if use_colors:
            prompt = """You are solving an ARC (Abstraction and Reasoning Corpus) task.
Below are the training examples and how humans solved similar transformations.

Colors: Black(0), Blue(1), Red(2), Green(3), Yellow(4), Gray(5), Pink(6), Orange(7), Purple(8), Brown(9).

Training Examples:
"""
        else:
            prompt = """You are solving an ARC (Abstraction and Reasoning Corpus) task.
Below are the training examples and how humans solved similar transformations.

Training Examples:
"""
        
        # 원본 예제들
        for i, pair in enumerate(task.train_pairs):
            prompt += f"\nExample {i+1}:\nInput:\n{self.grid_to_string(pair.x.tolist(), use_colors)}\n"
            prompt += f"Output:\n{self.grid_to_string(pair.y.tolist(), use_colors)}\n"
        
        # 인간의 풀이 과정 추가
        if action_traces and num_traces > 0:
            selected_traces = action_traces[:num_traces]
            prompt += f"\n\nHow humans solved this task (action sequences):"
            
            for i, trace in enumerate(selected_traces):
                simplified_trace = self.simplify_action_trace(trace)
                prompt += f"\n\nHuman Solution {i+1} ({len(trace)} steps):"
                
                # 처음 10개 action만 표시 (너무 길어지는 것 방지)
                for j, action in enumerate(simplified_trace[:10]):
                    prompt += f"\n  Step {j+1}: {action}"
                
                if len(simplified_trace) > 10:
                    prompt += f"\n  ... ({len(simplified_trace) - 10} more steps)"
                
                # 마지막 action 표시
                if simplified_trace[-1] != "Submit solution":
                    prompt += f"\n  Final step: {simplified_trace[-1]}"
        
        # 테스트 입력
        prompt += f"\n\nNow solve this test case:"
        prompt += f"\nTest Input:\n{self.grid_to_string(task.test_pairs[0].x.tolist(), use_colors)}\n"
        
        if use_colors:
            prompt += "\nProvide only the output grid in the same format (colors separated by spaces, rows on new lines):"
        else:
            prompt += "\nProvide only the output grid in the same format (numbers separated by spaces, rows on new lines):"
        
        return prompt
    
    def create_harc_hint_prompt(self, task: Dict, action_traces: List[List[Dict]], 
                               use_colors: bool = False) -> str:
        """Action trace를 힌트로 요약한 프롬프트"""
        if use_colors:
            prompt = """You are solving an ARC (Abstraction and Reasoning Corpus) task.

Colors: Black(0), Blue(1), Red(2), Green(3), Yellow(4), Gray(5), Pink(6), Orange(7), Purple(8), Brown(9).

Training Examples:
"""
        else:
            prompt = """You are solving an ARC (Abstraction and Reasoning Corpus) task.

Training Examples:
"""
        
        # 원본 예제들
        for i, pair in enumerate(task.train_pairs):
            prompt += f"\nExample {i+1}:\nInput:\n{self.grid_to_string(pair.x.tolist(), use_colors)}\n"
            prompt += f"Output:\n{self.grid_to_string(pair.y.tolist(), use_colors)}\n"
        
        # Action trace에서 패턴 추출
        if action_traces:
            # 가장 빈번한 action 패턴 찾기
            action_counts = defaultdict(int)
            for trace in action_traces:
                for action in trace:
                    action_counts[action['action']] += 1
            
            # 상위 action들
            top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            prompt += f"\n\nHint: Humans typically solved this using actions like: "
            prompt += ", ".join([action[0].replace('_', ' ') for action in top_actions])
            
            # 평균 스텝 수
            avg_steps = sum(len(trace) for trace in action_traces) / len(action_traces)
            prompt += f"\nAverage solution length: {avg_steps:.1f} steps"
        
        # 테스트 입력
        prompt += f"\n\nTest Input:\n{self.grid_to_string(task.test_pairs[0].x.tolist(), use_colors)}\n"
        
        if use_colors:
            prompt += "\nProvide only the output grid in the same format (colors separated by spaces, rows on new lines):"
        else:
            prompt += "\nProvide only the output grid in the same format (numbers separated by spaces, rows on new lines):"
        
        return prompt
    
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
    
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """LLM으로 응답 생성"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=2048).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                       skip_special_tokens=True)
        return response
    
    def extract_grid_from_response(self, response: str, from_colors: bool = False) -> List[List[int]]:
        """응답에서 그리드 추출"""
        lines = response.strip().split('\n')
        grid_lines = []
        
        if from_colors:
            color_pattern = '|'.join(COLOR_MAP.values())
            for line in lines:
                if re.search(color_pattern, line, re.IGNORECASE):
                    grid_lines.append(line.strip())
        else:
            for line in lines:
                if re.match(r'^[\d\s]+$', line.strip()) and line.strip():
                    grid_lines.append(line.strip())
        
        if grid_lines:
            return self.string_to_grid('\n'.join(grid_lines), from_colors)
        return []
    
    def evaluate_solution(self, predicted: List[List[int]], 
                         expected: List[List[int]]) -> bool:
        """예측과 정답 비교"""
        if not predicted or not expected:
            return False
        
        if len(predicted) != len(expected):
            return False
        
        for pred_row, exp_row in zip(predicted, expected):
            if len(pred_row) != len(exp_row):
                return False
            if pred_row != exp_row:
                return False
        
        return True
    
    # Action Sequence 관련 메서드들
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
    
    def run_harc_experiment(self, task_ids: List[str] = None,
                           prompt_types: List[str] = ['full_trace', 'hint'],
                           trace_counts: List[int] = [0, 1, 3, 5],
                           num_candidates: int = 5, use_colors: bool = False) -> Dict:
        """H-ARC 그리드 생성 실험 실행"""
        if task_ids is None:
            task_ids = ["6150a2bd", "87a80de6", "9ddd00f0", "d43fd935"]
        
        all_results = {}
        
        for task_id in task_ids:
            print(f"\n{'='*60}")
            print(f"Running H-ARC experiment on task: {task_id}")
            print(f"{'='*60}")
            
            # 태스크와 action trace 로드
            task = self.load_arc_task(task_id)
            if not task:
                print(f"Task {task_id} not found!")
                continue
            
            action_traces = self.load_harc_action_traces(task_id)
            
            task_results = {
                'task_id': task_id,
                'num_human_traces': len(action_traces),
                'results_by_prompt_type': {}
            }
            
            # 다양한 프롬프트 타입으로 실험
            for prompt_type in prompt_types:
                task_results['results_by_prompt_type'][prompt_type] = {}
                
                for num_traces in trace_counts:
                    if num_traces > len(action_traces):
                        continue
                    
                    print(f"\nTesting {prompt_type} prompt with {num_traces} human traces...")
                    
                    correct_count = 0
                    for _ in range(num_candidates):
                        if prompt_type == 'full_trace':
                            prompt = self.create_harc_prompt(task, action_traces, num_traces, use_colors)
                        else:  # hint
                            prompt = self.create_harc_hint_prompt(task, action_traces if num_traces > 0 else [], use_colors)
                        
                        response = self.generate_response(prompt, temperature=0.3)
                        predicted_grid = self.extract_grid_from_response(response, use_colors)
                        
                        expected_output = task.test_pairs[0].y.tolist()
                        is_correct = self.evaluate_solution(predicted_grid, expected_output)
                        if is_correct:
                            correct_count += 1
                    
                    accuracy = correct_count / num_candidates
                    task_results['results_by_prompt_type'][prompt_type][num_traces] = {
                        'accuracy': accuracy,
                        'correct_count': correct_count,
                        'total_candidates': num_candidates
                    }
                    
                    print(f"Accuracy with {prompt_type} prompt and {num_traces} traces: {accuracy:.2%}")
                    
                    if self.use_wandb:
                        wandb.log({
                            'task_id': task_id,
                            'prompt_type': prompt_type,
                            'num_traces': num_traces,
                            'accuracy': accuracy,
                            'experiment': 'h-arc_grid'
                        })
            
            all_results[task_id] = task_results
        
        return all_results
    
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
        
        return combined_results
    
    def save_results(self, results: Dict, output_file: str = "harc_experiment_results.json"):
        """결과 저장"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main():
    """H-ARC 실험 메인 함수"""
    experiment = HARCExperiment(use_wandb=True)
    
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