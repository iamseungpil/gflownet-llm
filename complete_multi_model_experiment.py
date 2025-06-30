"""
Complete Multi-Model ARC Experiment
모든 모델로 RE-ARC, H-ARC(그리드), H-ARC(액션) 실험을 통합 실행
"""

import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import re
import wandb
from tqdm import tqdm
from arc import train_problems, validation_problems
from collections import defaultdict
from difflib import SequenceMatcher
from openai import OpenAI
from abc import ABC, abstractmethod

# Color mapping for ARC grids
COLOR_MAP = {
    0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
    5: "Gray", 6: "Pink", 7: "Orange", 8: "Purple", 9: "Brown"
}

REVERSE_COLOR_MAP = {v: k for k, v in COLOR_MAP.items()}


class BaseARCModel(ABC):
    """모든 모델에 대한 기본 인터페이스"""
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass


class HuggingFaceModel(BaseARCModel):
    """Hugging Face 모델용 래퍼"""
    
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        print(f"Loading {model_name}...")
        cache_dir = os.getenv('HF_HOME', '/data/.cache/huggingface')
        print(f"Using cache directory: {cache_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        if "qwen" in model_name.lower():
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir
        )
        self.model.eval()
    
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
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
    
    def get_model_name(self) -> str:
        return self.model_name


class OpenAIModel(BaseARCModel):
    """OpenAI 모델용 래퍼"""
    
    def __init__(self, model_name: str = "o1-preview", api_key: str = None):
        self.model_name = model_name
        
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        print(f"Initialized OpenAI {model_name}")
    
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        try:
            if "o1" in self.model_name:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""
    
    def get_model_name(self) -> str:
        return f"openai/{self.model_name}"


class CompleteMultiModelExperiment:
    """모든 모델로 모든 실험을 수행하는 통합 클래스"""
    
    def __init__(self, models_config: Dict[str, Dict], use_wandb: bool = True):
        self.models = {}
        self.models_config = models_config
        
        # 모델 초기화
        for model_id, config in models_config.items():
            if config["type"] == "huggingface":
                self.models[model_id] = HuggingFaceModel(config["model_name"])
            elif config["type"] == "openai":
                self.models[model_id] = OpenAIModel(
                    config["model_name"], 
                    config.get("api_key")
                )
        
        # 데이터 경로 설정
        self.rearc_path = Path("data/re-arc/re_arc_extracted/re_arc/tasks")
        self.harc_data_path = Path("data/h-arc/data/data.csv")
        
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
                project="arc-complete-multi-model",
                config={
                    "models": list(models_config.keys()),
                    "experiment_type": "complete_multi_model"
                }
            )
    
    def load_arc_task(self, task_id: str):
        """원본 ARC 태스크 로드"""
        all_problems = train_problems + validation_problems
        
        for problem in all_problems:
            if problem.uid == task_id:
                return problem
        return None
    
    def load_rearc_augmentations(self, task_id: str) -> List[Dict]:
        """RE-ARC 증강 데이터 로드"""
        augmented_examples = []
        rearc_file = self.rearc_path / f"{task_id}.json"
        
        if rearc_file.exists():
            with open(rearc_file, 'r') as f:
                rearc_data = json.load(f)
                
                if isinstance(rearc_data, list):
                    augmented_examples = rearc_data
                elif isinstance(rearc_data, dict):
                    if 'train' in rearc_data:
                        augmented_examples = rearc_data.get('train', [])
                
                valid_examples = []
                for example in augmented_examples:
                    if isinstance(example, dict) and 'input' in example and 'output' in example:
                        valid_examples.append(example)
                
                augmented_examples = valid_examples[:20]  # 최대 20개만 사용
        
        return augmented_examples
    
    def load_harc_action_traces(self, task_id: str) -> List[List[Dict]]:
        """H-ARC action trace 로드"""
        if not self.harc_data_path.exists():
            print(f"H-ARC data file not found: {self.harc_data_path}")
            return []
        
        df = pd.read_csv(self.harc_data_path)
        task_data = df[df['task_name'] == task_id]
        
        if len(task_data) == 0:
            return []
        
        successful_attempts = task_data[task_data['solved'] == True]
        action_traces = []
        unique_attempts = successful_attempts[['hashed_id', 'attempt_number']].drop_duplicates()
        
        for _, row in unique_attempts.iterrows():
            participant_id = row['hashed_id']
            attempt_num = row['attempt_number']
            
            attempt_data = task_data[
                (task_data['hashed_id'] == participant_id) & 
                (task_data['attempt_number'] == attempt_num)
            ].sort_values('action_id')
            
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
            
            if actions:
                action_traces.append(actions)
        
        return action_traces
    
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
    
    def evaluate_solution(self, predicted: List[List[int]], expected: List[List[int]]) -> bool:
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
    
    # ============= RE-ARC 실험 메서드 =============
    
    def create_baseline_prompt(self, task: Dict, use_colors: bool = False) -> str:
        """기본 프롬프트 생성"""
        if use_colors:
            prompt = """You are solving an ARC task. Find the pattern and apply it to the test input.
Colors: Black(0), Blue(1), Red(2), Green(3), Yellow(4), Gray(5), Pink(6), Orange(7), Purple(8), Brown(9).

Training Examples:
"""
        else:
            prompt = """You are solving an ARC task. Find the pattern and apply it to the test input.

Training Examples:
"""
        
        for i, pair in enumerate(task.train_pairs):
            prompt += f"\nExample {i+1}:\nInput:\n{self.grid_to_string(pair.x.tolist(), use_colors)}\n"
            prompt += f"Output:\n{self.grid_to_string(pair.y.tolist(), use_colors)}\n"
        
        prompt += f"\nTest Input:\n{self.grid_to_string(task.test_pairs[0].x.tolist(), use_colors)}\n"
        prompt += "\nProvide only the output grid in the same format:"
        
        return prompt
    
    def create_augmented_prompt(self, task: Dict, augmented_examples: List[Dict], 
                               num_augmented: int = 5, use_colors: bool = False) -> str:
        """증강 데이터 포함 프롬프트"""
        prompt = self.create_baseline_prompt(task, use_colors)
        
        if augmented_examples and num_augmented > 0:
            selected_examples = augmented_examples[:num_augmented]
            prompt = prompt.replace("Training Examples:", "Original Training Examples:")
            
            additional_prompt = f"\nAdditional Examples:"
            for i, example in enumerate(selected_examples):
                additional_prompt += f"\nAugmented Example {i+1}:\nInput:\n{self.grid_to_string(example['input'], use_colors)}\n"
                additional_prompt += f"Output:\n{self.grid_to_string(example['output'], use_colors)}\n"
            
            # Test Input 앞에 추가 예제 삽입
            prompt = prompt.replace("\nTest Input:", additional_prompt + "\nTest Input:")
        
        return prompt
    
    # ============= H-ARC 그리드 실험 메서드 =============
    
    def simplify_action_trace(self, actions: List[Dict]) -> List[str]:
        """Action trace를 간단한 설명으로 변환"""
        simplified = []
        
        for action in actions:
            action_type = action['action']
            
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
            else:
                desc = action_type.replace('_', ' ').capitalize()
            
            simplified.append(desc)
        
        return simplified
    
    def create_harc_grid_prompt(self, task: Dict, action_traces: List[List[Dict]], 
                               num_traces: int = 3, prompt_type: str = "full_trace", 
                               use_colors: bool = False) -> str:
        """H-ARC 그리드 생성 프롬프트"""
        base_prompt = self.create_baseline_prompt(task, use_colors)
        
        if not action_traces or num_traces == 0:
            return base_prompt
        
        if prompt_type == "full_trace":
            # 상세한 action sequence 표시
            selected_traces = action_traces[:num_traces]
            trace_info = f"\n\nHow humans solved this task (action sequences):"
            
            for i, trace in enumerate(selected_traces):
                simplified_trace = self.simplify_action_trace(trace)
                trace_info += f"\n\nHuman Solution {i+1} ({len(trace)} steps):"
                
                for j, action in enumerate(simplified_trace[:10]):
                    trace_info += f"\n  Step {j+1}: {action}"
                
                if len(simplified_trace) > 10:
                    trace_info += f"\n  ... ({len(simplified_trace) - 10} more steps)"
        
        else:  # hint
            # 요약된 힌트만 표시
            action_counts = defaultdict(int)
            for trace in action_traces:
                for action in trace:
                    action_counts[action['action']] += 1
            
            top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            avg_steps = sum(len(trace) for trace in action_traces) / len(action_traces)
            
            trace_info = f"\n\nHint: Humans typically solved this using actions like: "
            trace_info += ", ".join([action[0].replace('_', ' ') for action in top_actions])
            trace_info += f"\nAverage solution length: {avg_steps:.1f} steps"
        
        # Test Input 앞에 trace 정보 삽입
        enhanced_prompt = base_prompt.replace("\nTest Input:", trace_info + "\n\nNow solve this test case:\nTest Input:")
        return enhanced_prompt
    
    # ============= H-ARC 액션 실험 메서드 =============
    
    def create_action_sequence_prompt(self, task: Dict, action_traces: List[List[Dict]], 
                                     num_examples: int = 3) -> str:
        """Action sequence 예측 프롬프트"""
        prompt = """You are analyzing how humans solve ARC tasks.
Given the training examples, predict the sequence of actions needed to solve the test case.

Available actions:
- click_cell, select_color, flood_fill, copy, paste, resize_grid, rotate, flip, submit

Training Examples:
"""
        
        for i, pair in enumerate(task.train_pairs):
            prompt += f"\nExample {i+1}:\nInput:\n{self.grid_to_string(pair.x.tolist())}\n"
            prompt += f"Output:\n{self.grid_to_string(pair.y.tolist())}\n"
        
        if action_traces and num_examples > 0:
            prompt += "\n\nHow humans solved this task (example action sequences):"
            
            for i, trace in enumerate(action_traces[:num_examples]):
                prompt += f"\n\nHuman Solution {i+1}:"
                
                for j, action in enumerate(trace):
                    action_type = action['action']
                    prompt += f"\n{j+1}. {action_type}"
                    
                    if action.get('selected_symbol') is not None:
                        prompt += f" (color: {action['selected_symbol']})"
        
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
            if re.match(r'^\d+\.', line):
                action = re.sub(r'^\d+\.\s*', '', line)
                action = re.sub(r'\s*\([^)]*\)', '', action)
                action = action.strip().lower().replace(' ', '_')
                
                normalized_action = self.normalize_action(action)
                if normalized_action:
                    actions.append(normalized_action)
        
        return actions
    
    def normalize_action(self, action: str) -> str:
        """Action 이름 정규화"""
        action_lower = action.lower().strip()
        
        for standard_action, variants in self.action_mapping.items():
            if action_lower == standard_action:
                return standard_action
            for variant in variants:
                if variant in action_lower or action_lower in variant:
                    return standard_action
        
        return action
    
    def evaluate_action_sequence(self, predicted: List[str], actual_traces: List[List[Dict]]) -> Dict:
        """Action sequence 평가"""
        if not predicted or not actual_traces:
            return {
                'exact_match': 0, 
                'similarity': 0, 
                'common_actions_ratio': 0,
                'predicted_length': len(predicted) if predicted else 0
            }
        
        actual_sequences = []
        for trace in actual_traces:
            sequence = [self.normalize_action(action['action']) for action in trace]
            actual_sequences.append(sequence)
        
        max_similarity = 0
        best_match_info = {}
        
        for actual_seq in actual_sequences:
            if predicted == actual_seq:
                return {
                    'exact_match': 1.0,
                    'similarity': 1.0,
                    'common_actions_ratio': 1.0,
                    'predicted_length': len(predicted),
                    'actual_length': len(actual_seq),
                    'match_type': 'exact'
                }
            
            matcher = SequenceMatcher(None, predicted, actual_seq)
            similarity = matcher.ratio()
            
            common = set(predicted) & set(actual_seq)
            common_ratio = len(common) / max(len(set(predicted)), len(set(actual_seq))) if max(len(set(predicted)), len(set(actual_seq))) > 0 else 0
            
            combined_similarity = (similarity + common_ratio) / 2
            
            if combined_similarity > max_similarity:
                max_similarity = combined_similarity
                best_match_info = {
                    'exact_match': 0.0,
                    'similarity': similarity,
                    'common_actions_ratio': common_ratio,
                    'combined_similarity': combined_similarity,
                    'predicted_length': len(predicted),
                    'actual_length': len(actual_seq),
                    'match_type': 'partial'
                }
        
        return best_match_info
    
    # ============= 통합 실험 실행 메서드 =============
    
    def run_complete_experiment(self, task_ids: List[str] = None,
                               augmentation_sizes: List[int] = [0, 5, 10],
                               harc_trace_counts: List[int] = [0, 3],
                               harc_action_examples: List[int] = [0, 3],
                               num_candidates: int = 3) -> Dict:
        """모든 모델로 모든 실험 실행"""
        if task_ids is None:
            task_ids = ["74dd1130"]
        
        all_results = {
            'experiment_summary': {
                'models_tested': list(self.models.keys()),
                'tasks_tested': task_ids,
                'experiment_types': ['re-arc', 'h-arc-grid', 'h-arc-action']
            },
            'detailed_results': {}
        }
        
        for task_id in task_ids:
            print(f"\n{'='*80}")
            print(f"COMPLETE EXPERIMENT ON TASK: {task_id}")
            print(f"{'='*80}")
            
            # 데이터 로드
            task = self.load_arc_task(task_id)
            if not task:
                print(f"Task {task_id} not found!")
                continue
            
            augmented_examples = self.load_rearc_augmentations(task_id)
            action_traces = self.load_harc_action_traces(task_id)
            
            task_results = {
                'task_id': task_id,
                'num_augmented_available': len(augmented_examples),
                'num_action_traces': len(action_traces),
                'model_results': {}
            }
            
            # 각 모델에 대해 모든 실험 수행
            for model_id, model in self.models.items():
                print(f"\n{'-'*60}")
                print(f"Testing Model: {model_id}")
                print(f"{'-'*60}")
                
                model_results = {
                    're-arc': {},
                    'h-arc-grid': {},
                    'h-arc-action': {}
                }
                
                # 1. RE-ARC 실험
                print("\n1. RE-ARC Experiment")
                for num_aug in augmentation_sizes:
                    if num_aug > len(augmented_examples):
                        continue
                    
                    print(f"  Testing with {num_aug} augmented examples...")
                    
                    if num_aug == 0:
                        prompt = self.create_baseline_prompt(task, use_colors=False)
                    else:
                        prompt = self.create_augmented_prompt(task, augmented_examples, num_aug, use_colors=False)
                    
                    correct_count = 0
                    responses_and_predictions = []
                    for _ in range(num_candidates):
                        response = model.generate_response(prompt, temperature=0.3)
                        predicted_grid = self.extract_grid_from_response(response, from_colors=False)
                        expected_output = task.test_pairs[0].y.tolist()
                        
                        is_correct = self.evaluate_solution(predicted_grid, expected_output)
                        if is_correct:
                            correct_count += 1
                        
                        responses_and_predictions.append({
                            'prompt': prompt,
                            'response': response,
                            'predicted_grid': predicted_grid,
                            'expected_grid': expected_output,
                            'is_correct': is_correct
                        })
                    
                    accuracy = correct_count / num_candidates
                    model_results['re-arc'][num_aug] = {
                        'accuracy': accuracy,
                        'correct_count': correct_count,
                        'total': num_candidates,
                        'responses_and_predictions': responses_and_predictions
                    }
                    print(f"    Accuracy: {accuracy:.2%}")
                
                # 2. H-ARC 그리드 실험
                print("\n2. H-ARC Grid Generation Experiment")
                for prompt_type in ['full_trace', 'hint']:
                    model_results['h-arc-grid'][prompt_type] = {}
                    
                    for num_traces in harc_trace_counts:
                        if num_traces > len(action_traces):
                            continue
                        
                        print(f"  Testing {prompt_type} with {num_traces} traces...")
                        
                        prompt = self.create_harc_grid_prompt(task, action_traces, num_traces, prompt_type, use_colors=False)
                        
                        correct_count = 0
                        responses_and_predictions = []
                        for _ in range(num_candidates):
                            response = model.generate_response(prompt, temperature=0.3)
                            predicted_grid = self.extract_grid_from_response(response, from_colors=False)
                            expected_output = task.test_pairs[0].y.tolist()
                            
                            is_correct = self.evaluate_solution(predicted_grid, expected_output)
                            if is_correct:
                                correct_count += 1
                            
                            responses_and_predictions.append({
                                'prompt': prompt,
                                'response': response,
                                'predicted_grid': predicted_grid,
                                'expected_grid': expected_output,
                                'is_correct': is_correct
                            })
                        
                        accuracy = correct_count / num_candidates
                        model_results['h-arc-grid'][prompt_type][num_traces] = {
                            'accuracy': accuracy,
                            'correct_count': correct_count,
                            'total': num_candidates,
                            'responses_and_predictions': responses_and_predictions
                        }
                        print(f"    Accuracy: {accuracy:.2%}")
                
                # 3. H-ARC 액션 실험
                print("\n3. H-ARC Action Sequence Experiment")
                if action_traces:
                    for num_examples in harc_action_examples:
                        if num_examples > len(action_traces):
                            continue
                        
                        print(f"  Testing with {num_examples} action examples...")
                        
                        evaluation_results = []
                        responses_and_predictions = []
                        
                        for _ in range(num_candidates):
                            prompt = self.create_action_sequence_prompt(task, action_traces, num_examples)
                            response = model.generate_response(prompt, max_tokens=512, temperature=0.3)
                            predicted_actions = self.parse_predicted_actions(response)
                            
                            eval_result = self.evaluate_action_sequence(predicted_actions, action_traces)
                            evaluation_results.append(eval_result)
                            
                            responses_and_predictions.append({
                                'prompt': prompt,
                                'response': response,
                                'predicted_actions': predicted_actions,
                                'evaluation': eval_result
                            })
                        
                        avg_similarity = sum(r.get('combined_similarity', r.get('similarity', 0)) 
                                           for r in evaluation_results) / len(evaluation_results)
                        exact_matches = sum(r.get('exact_match', 0) for r in evaluation_results)
                        avg_common_ratio = sum(r.get('common_actions_ratio', 0) 
                                             for r in evaluation_results) / len(evaluation_results)
                        
                        model_results['h-arc-action'][num_examples] = {
                            'avg_similarity': avg_similarity,
                            'avg_common_actions_ratio': avg_common_ratio,
                            'exact_match_rate': exact_matches / num_candidates,
                            'total': num_candidates,
                            'responses_and_predictions': responses_and_predictions
                        }
                        print(f"    Avg Similarity: {avg_similarity:.3f}")
                        print(f"    Exact Match Rate: {exact_matches}/{num_candidates}")
                else:
                    print("  No action traces available for this task")
                
                task_results['model_results'][model_id] = model_results
                
                # W&B 로깅
                if self.use_wandb:
                    wandb.log({
                        'task_id': task_id,
                        'model_id': model_id,
                        're-arc_baseline': model_results['re-arc'].get(0, {}).get('accuracy', 0),
                        're-arc_best': max([r.get('accuracy', 0) for r in model_results['re-arc'].values()]) if model_results['re-arc'] else 0,
                        'h-arc-grid_best': max([max([r.get('accuracy', 0) for r in pt.values()]) for pt in model_results['h-arc-grid'].values()]) if model_results['h-arc-grid'] else 0,
                        'h-arc-action_best': max([r.get('avg_similarity', 0) for r in model_results['h-arc-action'].values()]) if model_results['h-arc-action'] else 0
                    })
            
            all_results['detailed_results'][task_id] = task_results
        
        # 종합 요약 생성
        self.generate_comprehensive_summary(all_results)
        
        return all_results
    
    def generate_comprehensive_summary(self, results: Dict):
        """종합 요약 생성"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EXPERIMENT SUMMARY")
        print("="*80)
        
        models = results['experiment_summary']['models_tested']
        tasks = results['experiment_summary']['tasks_tested']
        
        # 모델별 평균 성능 계산
        model_summary = {}
        
        for model_id in models:
            re_arc_scores = []
            h_arc_grid_scores = []
            h_arc_action_scores = []
            
            for task_id in tasks:
                if task_id in results['detailed_results']:
                    task_data = results['detailed_results'][task_id]
                    if model_id in task_data['model_results']:
                        model_data = task_data['model_results'][model_id]
                        
                        # RE-ARC 점수
                        if model_data['re-arc']:
                            re_arc_scores.extend([r.get('accuracy', 0) for r in model_data['re-arc'].values()])
                        
                        # H-ARC Grid 점수
                        if model_data['h-arc-grid']:
                            for prompt_type in model_data['h-arc-grid'].values():
                                h_arc_grid_scores.extend([r.get('accuracy', 0) for r in prompt_type.values()])
                        
                        # H-ARC Action 점수
                        if model_data['h-arc-action']:
                            h_arc_action_scores.extend([r.get('avg_similarity', 0) for r in model_data['h-arc-action'].values()])
            
            model_summary[model_id] = {
                're-arc_avg': np.mean(re_arc_scores) if re_arc_scores else 0,
                're-arc_max': max(re_arc_scores) if re_arc_scores else 0,
                'h-arc-grid_avg': np.mean(h_arc_grid_scores) if h_arc_grid_scores else 0,
                'h-arc-grid_max': max(h_arc_grid_scores) if h_arc_grid_scores else 0,
                'h-arc-action_avg': np.mean(h_arc_action_scores) if h_arc_action_scores else 0,
                'h-arc-action_max': max(h_arc_action_scores) if h_arc_action_scores else 0
            }
        
        # 결과 출력
        print(f"\n{'Model':<15} {'RE-ARC':<15} {'H-ARC Grid':<15} {'H-ARC Action':<15}")
        print(f"{'='*15} {'='*15} {'='*15} {'='*15}")
        
        for model_id, summary in model_summary.items():
            print(f"{model_id:<15} "
                  f"{summary['re-arc_max']:.1%}({summary['re-arc_avg']:.1%})<15 "
                  f"{summary['h-arc-grid_max']:.1%}({summary['h-arc-grid_avg']:.1%})<15 "
                  f"{summary['h-arc-action_max']:.3f}({summary['h-arc-action_avg']:.3f})<15")
        
        print(f"\nNote: Format is MAX(AVG) for each experiment type")
        
        # 최고 성능 모델 찾기
        best_re_arc = max(models, key=lambda m: model_summary[m]['re-arc_max'])
        best_h_arc_grid = max(models, key=lambda m: model_summary[m]['h-arc-grid_max'])
        best_h_arc_action = max(models, key=lambda m: model_summary[m]['h-arc-action_max'])
        
        print(f"\nBest Performing Models:")
        print(f"  RE-ARC: {best_re_arc} ({model_summary[best_re_arc]['re-arc_max']:.1%})")
        print(f"  H-ARC Grid: {best_h_arc_grid} ({model_summary[best_h_arc_grid]['h-arc-grid_max']:.1%})")
        print(f"  H-ARC Action: {best_h_arc_action} ({model_summary[best_h_arc_action]['h-arc-action_max']:.3f})")
        
        results['model_summary'] = model_summary
        results['best_models'] = {
            'best_re_arc': best_re_arc,
            'best_h_arc_grid': best_h_arc_grid,
            'best_h_arc_action': best_h_arc_action
        }
    
    def save_results(self, results: Dict, output_file: str = "complete_multi_model_results.json"):
        """결과 저장"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def run_quick_test():
    """빠른 테스트"""
    models_config = {
        "llama3.1-8b": {
            "type": "huggingface",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct"
        }
    }
    
    experiment = CompleteMultiModelExperiment(models_config, use_wandb=False)
    
    results = experiment.run_complete_experiment(
        task_ids=["74dd1130"],
        augmentation_sizes=[0, 5],
        harc_trace_counts=[0, 3],
        harc_action_examples=[0, 3],
        num_candidates=2
    )
    
    experiment.save_results(results, "quick_complete_test.json")
    print("Quick test completed!")


def get_models_config(include_openai: bool = True, include_gemma: bool = True, include_qwen: bool = True):
    """모델 설정을 선택적으로 생성"""
    models_config = {
        "llama3.1-8b": {
            "type": "huggingface",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct"
        }
    }
    
    if include_openai:
        models_config["openai-o1"] = {
            "type": "openai",
            "model_name": "o1-preview",
            "api_key": ""
        }
    
    if include_gemma:
        models_config["gemma-7b"] = {
            "type": "huggingface",
            "model_name": "google/gemma-7b-it"
        }
    
    if include_qwen:
        models_config["qwen2.5-7b"] = {
            "type": "huggingface",
            "model_name": "Qwen/Qwen2.5-7B-Instruct"
        }
    
    return models_config


def print_usage():
    """사용법 출력"""
    print("""
Usage: python complete_multi_model_experiment.py [options]

Options:
  quick                 Run quick test (Llama only, 1 task, 2 candidates)
  --models MODEL1,MODEL2,...  Specify models to run (comma-separated)
                       Available: llama3.1-8b, openai-o1, gemma-7b, qwen2.5-7b
  --no-openai          Exclude OpenAI models (avoid API costs)
  --no-gemma           Exclude Gemma model
  --no-qwen            Exclude Qwen model
  --llama-only         Run with Llama 3.1-8B only
  --help               Show this help message

Examples:
  python complete_multi_model_experiment.py quick
  python complete_multi_model_experiment.py --models llama3.1-8b,gemma-7b
  python complete_multi_model_experiment.py --no-openai
  python complete_multi_model_experiment.py --llama-only
  python complete_multi_model_experiment.py --no-openai --no-gemma
""")


def main():
    """메인 실행 함수"""
    import sys
    
    # 명령행 인수 파싱
    if "--help" in sys.argv or "-h" in sys.argv:
        print_usage()
        return
    
    if "quick" in sys.argv:
        run_quick_test()
        return
    
    # --models 옵션 확인
    specific_models = None
    for i, arg in enumerate(sys.argv):
        if arg == "--models" and i + 1 < len(sys.argv):
            specific_models = sys.argv[i + 1].split(',')
            break
    
    if specific_models:
        # 특정 모델만 선택
        all_models_config = get_models_config(True, True, True)
        models_config = {}
        for model_id in specific_models:
            if model_id in all_models_config:
                models_config[model_id] = all_models_config[model_id]
            else:
                print(f"Warning: Unknown model '{model_id}'. Available models: {', '.join(all_models_config.keys())}")
        
        if not models_config:
            print("Error: No valid models specified.")
            return
    else:
        # 기존 로직: 제외 옵션 사용
        include_openai = "--no-openai" not in sys.argv
        include_gemma = "--no-gemma" not in sys.argv
        include_qwen = "--no-qwen" not in sys.argv
        
        if "--llama-only" in sys.argv:
            include_openai = False
            include_gemma = False
            include_qwen = False
        
        # 모델 설정 생성
        models_config = get_models_config(include_openai, include_gemma, include_qwen)
    
    # 선택된 모델 출력
    print(f"\n🚀 Starting Complete Multi-Model Experiment")
    print(f"Selected models: {', '.join(models_config.keys())}")
    
    # OpenAI 모델이 포함되었는지 확인
    has_openai = any('openai' in model_id for model_id in models_config.keys())
    if not has_openai:
        print("💰 OpenAI models excluded (no API costs)")
    
    print(f"📊 Testing {len(models_config)} models on 4 tasks with 3 experiment types")
    estimated_time = len(models_config) * 60  # 모델당 약 60분
    print(f"⏱️  Estimated time: ~{estimated_time//60}h {estimated_time%60}m")
    print()
    
    # 실험 초기화
    experiment = CompleteMultiModelExperiment(models_config, use_wandb=True)
    
    # 완전한 실험 실행
    results = experiment.run_complete_experiment(
        task_ids=["74dd1130"],
        augmentation_sizes=[0, 5, 10],
        harc_trace_counts=[0, 3, 5],
        harc_action_examples=[0, 3],
        num_candidates=5
    )
    
    # 결과 저장
    experiment.save_results(results)
    
    if experiment.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()