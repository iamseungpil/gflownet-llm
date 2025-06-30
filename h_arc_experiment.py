"""
H-ARC Experiment  
H-ARC 데이터를 사용한 ARC 태스크 해결 실험 (그리드 생성 및 액션 시퀀스 예측)
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
    
    def generate_response(self, prompt, max_tokens: int = 512, temperature: float = 0.1) -> str:
        if isinstance(prompt, list):
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    prompt, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                if "System role not supported" in str(e):
                    # System role을 지원하지 않는 모델의 경우 user 메시지로 합치기
                    combined_content = ""
                    for msg in prompt:
                        if msg["role"] == "system":
                            combined_content += f"Instructions: {msg['content']}\n\n"
                        elif msg["role"] == "user":
                            combined_content += msg["content"]
                    
                    user_only_prompt = [{"role": "user", "content": combined_content}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        user_only_prompt,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    raise e
        else:
            formatted_prompt = prompt
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, 
                               max_length=2048).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
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
    
    def generate_response(self, prompt, max_tokens: int = 512, temperature: float = 0.1) -> str:
        try:
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": prompt}]
            
            if "o1" in self.model_name:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=max_tokens
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""
    
    def get_model_name(self) -> str:
        return f"openai/{self.model_name}"


class HARCExperiment:
    """H-ARC 실험을 위한 클래스"""
    
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
                project="arc-h-arc-experiment",
                config={
                    "models": list(models_config.keys()),
                    "experiment_type": "h-arc"
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
        
        if from_colors:
            color_pattern = '|'.join(COLOR_MAP.values())
            grid_lines = []
            
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                    
                if re.search(color_pattern, line, re.IGNORECASE):
                    normalized_line = line
                    for color in COLOR_MAP.values():
                        normalized_line = re.sub(rf'\b{re.escape(color)}\b', color, normalized_line, flags=re.IGNORECASE)
                    grid_lines.insert(0, normalized_line)
                elif grid_lines:
                    break
            
            if len(grid_lines) >= 2:
                return self.string_to_grid('\n'.join(grid_lines), from_colors)
        else:
            grid_lines = []
            for line in reversed(lines):
                line = line.strip()
                if re.match(r'^[\d\s]+$', line) and line:
                    grid_lines.insert(0, line)
                elif grid_lines:
                    break
            
            if len(grid_lines) >= 2:
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
                               use_colors: bool = False) -> List[Dict[str, str]]:
        """H-ARC 그리드 생성 프롬프트"""
        system_content = "You are an world-class puzzle solver who are extremely good at spotting patterns and solving puzzles. You are also an expert Python programmer who can write code to solve puzzles."
        
        examples_text = ""
        for i, pair in enumerate(task.train_pairs, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Input:\n{self.grid_to_string(pair.x.tolist(), use_colors)}\n"
            examples_text += f"Output:\n{self.grid_to_string(pair.y.tolist(), use_colors)}\n\n"
        
        trace_info = ""
        if action_traces and num_traces > 0:
            if prompt_type == "full_trace":
                selected_traces = action_traces[:num_traces]
                trace_info = "\n\nHow humans solved this task (action sequences):"
                
                for i, trace in enumerate(selected_traces):
                    simplified_trace = self.simplify_action_trace(trace)
                    trace_info += f"\n\nHuman Solution {i+1} ({len(trace)} steps):"
                    
                    for j, action in enumerate(simplified_trace[:10]):
                        trace_info += f"\n  Step {j+1}: {action}"
                    
                    if len(simplified_trace) > 10:
                        trace_info += f"\n  ... ({len(simplified_trace) - 10} more steps)"
            
            else:  # hint
                action_counts = defaultdict(int)
                for trace in action_traces:
                    for action in trace:
                        action_counts[action['action']] += 1
                
                top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                avg_steps = sum(len(trace) for trace in action_traces) / len(action_traces)
                
                trace_info = f"\n\nHint: Humans typically solved this using actions like: "
                trace_info += ", ".join([action[0].replace('_', ' ') for action in top_actions])
                trace_info += f"\nAverage solution length: {avg_steps:.1f} steps"
        
        if use_colors:
            color_info = "Each grid is represented as a 2D array where each cell is represented by a color. The available colors are: Black, Blue, Red, Green, Yellow, Gray, Pink, Orange, Purple, Brown."
        else:
            color_info = "Each grid is represented as a 2D array where each cell is represented by a number from 0-9."
            
        user_content = f"""The following is a puzzle from the ARC dataset. Given training examples of input and output grids, predict the output grid for the test inputs.
        {color_info} The grid input and output are written as a string where each cell is separated by a space and each row is separated by a newline.
        Here are the input and output grids for the training examples:
        {examples_text.strip()}{trace_info}

        Now solve this test case:
        Input:
        {self.grid_to_string(task.test_pairs[0].x.tolist(), use_colors)}

        Please provide ONLY the output grid in the exact same format as the examples. Do not include any explanation, code, or additional text - just the color grid."""
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
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
    
    def run_experiment(self, task_ids: List[str] = None,
                      harc_trace_counts: List[int] = [0, 3],
                      harc_action_examples: List[int] = [0, 3],
                      num_candidates: int = 3) -> Dict:
        """H-ARC 실험 실행"""
        if task_ids is None:
            task_ids = ["74dd1130"]
        
        all_results = {
            'experiment_summary': {
                'models_tested': list(self.models.keys()),
                'tasks_tested': task_ids,
                'experiment_type': 'h-arc'
            },
            'detailed_results': {}
        }
        
        for task_id in task_ids:
            print(f"\n{'='*80}")
            print(f"H-ARC EXPERIMENT ON TASK: {task_id}")
            print(f"{'='*80}")
            
            task = self.load_arc_task(task_id)
            if not task:
                print(f"Task {task_id} not found!")
                continue
            
            action_traces = self.load_harc_action_traces(task_id)
            
            task_results = {
                'task_id': task_id,
                'num_action_traces': len(action_traces),
                'model_results': {}
            }
            
            for model_id, model in self.models.items():
                print(f"\n{'-'*60}")
                print(f"Testing Model: {model_id}")
                print(f"{'-'*60}")
                
                model_results = {
                    'grid_generation': {},
                    'action_sequence': {}
                }
                
                # 1. H-ARC 그리드 생성 실험
                print("\n1. H-ARC Grid Generation Experiment")
                for prompt_type in ['full_trace', 'hint']:
                    model_results['grid_generation'][prompt_type] = {}
                    
                    for num_traces in harc_trace_counts:
                        if num_traces > len(action_traces):
                            continue
                        
                        print(f"  Testing {prompt_type} with {num_traces} traces...")
                        
                        prompt = self.create_harc_grid_prompt(task, action_traces, num_traces, prompt_type, use_colors=True)
                        
                        correct_count = 0
                        responses_and_predictions = []
                        for _ in range(num_candidates):
                            response = model.generate_response(prompt, max_tokens=100, temperature=0.1)
                            predicted_grid = self.extract_grid_from_response(response, from_colors=True)
                            expected_output = task.test_pairs[0].y.tolist()
                            
                            is_correct = self.evaluate_solution(predicted_grid, expected_output)
                            if is_correct:
                                correct_count += 1
                            
                            responses_and_predictions.append({
                                'response': response,
                                'predicted_grid': predicted_grid,
                                'expected_grid': expected_output,
                                'is_correct': is_correct
                            })
                        
                        accuracy = correct_count / num_candidates
                        model_results['grid_generation'][prompt_type][num_traces] = {
                            'accuracy': accuracy,
                            'correct_count': correct_count,
                            'total': num_candidates,
                            'responses_and_predictions': responses_and_predictions
                        }
                        print(f"    Accuracy: {accuracy:.2%}")
                
                # 2. H-ARC 액션 시퀀스 실험
                print("\n2. H-ARC Action Sequence Experiment")
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
                                'response': response,
                                'predicted_actions': predicted_actions,
                                'evaluation': eval_result
                            })
                        
                        avg_similarity = sum(r.get('combined_similarity', r.get('similarity', 0)) 
                                           for r in evaluation_results) / len(evaluation_results)
                        exact_matches = sum(r.get('exact_match', 0) for r in evaluation_results)
                        avg_common_ratio = sum(r.get('common_actions_ratio', 0) 
                                             for r in evaluation_results) / len(evaluation_results)
                        
                        model_results['action_sequence'][num_examples] = {
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
                        'grid_best_accuracy': max([max([r.get('accuracy', 0) for r in pt.values()]) for pt in model_results['grid_generation'].values()]) if model_results['grid_generation'] else 0,
                        'action_best_similarity': max([r.get('avg_similarity', 0) for r in model_results['action_sequence'].values()]) if model_results['action_sequence'] else 0
                    })
            
            all_results['detailed_results'][task_id] = task_results
        
        # 종합 요약 생성
        self.generate_summary(all_results)
        return all_results
    
    def generate_summary(self, results: Dict):
        """실험 결과 요약"""
        print("\n" + "="*80)
        print("H-ARC EXPERIMENT SUMMARY")
        print("="*80)
        
        models = results['experiment_summary']['models_tested']
        tasks = results['experiment_summary']['tasks_tested']
        
        model_summary = {}
        for model_id in models:
            grid_accuracies = []
            action_similarities = []
            
            for task_id in tasks:
                if task_id in results['detailed_results']:
                    task_data = results['detailed_results'][task_id]
                    if model_id in task_data['model_results']:
                        model_data = task_data['model_results'][model_id]
                        
                        # Grid generation 점수
                        if model_data['grid_generation']:
                            for prompt_type in model_data['grid_generation'].values():
                                grid_accuracies.extend([r.get('accuracy', 0) for r in prompt_type.values()])
                        
                        # Action sequence 점수
                        if model_data['action_sequence']:
                            action_similarities.extend([r.get('avg_similarity', 0) for r in model_data['action_sequence'].values()])
            
            model_summary[model_id] = {
                'grid_avg_accuracy': np.mean(grid_accuracies) if grid_accuracies else 0,
                'grid_max_accuracy': max(grid_accuracies) if grid_accuracies else 0,
                'action_avg_similarity': np.mean(action_similarities) if action_similarities else 0,
                'action_max_similarity': max(action_similarities) if action_similarities else 0
            }
        
        print(f"\n{'Model':<20} {'Grid Gen':<15} {'Action Seq':<15}")
        print(f"{'='*20} {'='*15} {'='*15}")
        
        for model_id, summary in model_summary.items():
            print(f"{model_id:<20} {summary['grid_max_accuracy']:.1%}({summary['grid_avg_accuracy']:.1%})<15 {summary['action_max_similarity']:.3f}({summary['action_avg_similarity']:.3f})<15")
        
        print(f"\nNote: Format is MAX(AVG) for each experiment type")
        
        best_grid_model = max(models, key=lambda m: model_summary[m]['grid_max_accuracy'])
        best_action_model = max(models, key=lambda m: model_summary[m]['action_max_similarity'])
        
        print(f"\nBest Performing Models:")
        print(f"  Grid Generation: {best_grid_model} ({model_summary[best_grid_model]['grid_max_accuracy']:.1%})")
        print(f"  Action Sequence: {best_action_model} ({model_summary[best_action_model]['action_max_similarity']:.3f})")
        
        results['model_summary'] = model_summary
        results['best_models'] = {
            'best_grid_model': best_grid_model,
            'best_action_model': best_action_model
        }
    
    def save_results(self, results: Dict, output_file: str = "h_arc_results.json"):
        """결과 저장"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def get_models_config(include_openai: bool = True, include_gemma: bool = True, include_qwen: bool = True):
    """모델 설정 생성"""
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


def main():
    """메인 실행 함수"""
    import sys
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
H-ARC Experiment

Usage: python h_arc_experiment.py [options]

Options:
  --models MODEL1,MODEL2,...  Specify models to run (comma-separated)
                       Available: llama3.1-8b, openai-o1, gemma-7b, qwen2.5-7b
  --no-openai          Exclude OpenAI models
  --no-gemma           Exclude Gemma model  
  --no-qwen            Exclude Qwen model
  --llama-only         Run with Llama 3.1-8B only
  --help               Show this help message
""")
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
        
        models_config = get_models_config(include_openai, include_gemma, include_qwen)
    
    print(f"🚀 Starting H-ARC Experiment")
    print(f"Selected models: {', '.join(models_config.keys())}")
    
    experiment = HARCExperiment(models_config, use_wandb=True)
    
    results = experiment.run_experiment(
        task_ids=["74dd1130", "1190e5a7", "150deff5", "178fcbfb"],
        harc_trace_counts=[0, 3, 5],
        harc_action_examples=[0, 3],
        num_candidates=3
    )
    
    experiment.save_results(results)
    
    if experiment.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()