"""
RE-ARC Experiment
RE-ARC 데이터를 사용한 ARC 태스크 해결 실험
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


class REARCExperiment:
    """RE-ARC 실험을 위한 클래스"""
    
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
        
        # W&B 초기화
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
            wandb.init(
                project="arc-re-arc-experiment",
                config={
                    "models": list(models_config.keys()),
                    "experiment_type": "re-arc"
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
    
    def create_baseline_prompt(self, task: Dict, use_colors: bool = False) -> List[Dict[str, str]]:
        """기본 프롬프트 생성"""
        system_content = "You are an world-class puzzle solver who are extremely good at spotting patterns and solving puzzles. You are also an expert Python programmer who can write code to solve puzzles."
        
        examples_text = ""
        for i, pair in enumerate(task.train_pairs, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Input:\n{self.grid_to_string(pair.x.tolist(), use_colors)}\n"
            examples_text += f"Output:\n{self.grid_to_string(pair.y.tolist(), use_colors)}\n\n"
        
        if use_colors:
            color_info = "Each grid is represented as a 2D array where each cell is represented by a color. The available colors are: Black, Blue, Red, Green, Yellow, Gray, Pink, Orange, Purple, Brown."
        else:
            color_info = "Each grid is represented as a 2D array where each cell is represented by a number from 0-9."
            
        user_content = f"""The following is a puzzle from the ARC dataset. Given training examples of input and output grids, predict the output grid for the test inputs.
        {color_info} The grid input and output are written as a string where each cell is separated by a space and each row is separated by a newline.
        Here are the input and output grids for the training examples:
        {examples_text.strip()}

        Here are the input grids for the test example:
        Input:
        {self.grid_to_string(task.test_pairs[0].x.tolist(), use_colors)}

        Please provide ONLY the output grid in the exact same format as the examples. Do not include any explanation, code, or additional text - just the color grid."""
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    def create_augmented_prompt(self, task: Dict, augmented_examples: List[Dict], 
                               num_augmented: int = 5, use_colors: bool = False) -> List[Dict[str, str]]:
        """증강 데이터 포함 프롬프트"""
        system_content = "You are an world-class puzzle solver who are extremely good at spotting patterns and solving puzzles. You are also an expert Python programmer who can write code to solve puzzles."
        
        examples_text = ""
        for i, pair in enumerate(task.train_pairs, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Input:\n{self.grid_to_string(pair.x.tolist(), use_colors)}\n"
            examples_text += f"Output:\n{self.grid_to_string(pair.y.tolist(), use_colors)}\n\n"
        
        if augmented_examples and num_augmented > 0:
            selected_examples = augmented_examples[:num_augmented]
            additional_text = "\nAdditional Examples:\n"
            for i, example in enumerate(selected_examples, 1):
                additional_text += f"Augmented Example {i}:\n"
                additional_text += f"Input:\n{self.grid_to_string(example['input'], use_colors)}\n"
                additional_text += f"Output:\n{self.grid_to_string(example['output'], use_colors)}\n\n"
            examples_text += additional_text
        
        if use_colors:
            color_info = "Each grid is represented as a 2D array where each cell is represented by a color. The available colors are: Black, Blue, Red, Green, Yellow, Gray, Pink, Orange, Purple, Brown."
        else:
            color_info = "Each grid is represented as a 2D array where each cell is represented by a number from 0-9."
            
        user_content = f"""The following is a puzzle from the ARC dataset. Given training examples of input and output grids, predict the output grid for the test inputs.
        {color_info} The grid input and output are written as a string where each cell is separated by a space and each row is separated by a newline.
        Here are the input and output grids for the training examples:
        {examples_text.strip()}

        Here are the input grids for the test example:
        Input:
        {self.grid_to_string(task.test_pairs[0].x.tolist(), use_colors)}

        Please provide ONLY the output grid in the exact same format as the examples. Do not include any explanation, code, or additional text - just the color grid."""
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    def run_experiment(self, task_ids: List[str] = None,
                      augmentation_sizes: List[int] = [0, 5, 10],
                      num_candidates: int = 3) -> Dict:
        """RE-ARC 실험 실행"""
        if task_ids is None:
            task_ids = ["74dd1130"]
        
        all_results = {
            'experiment_summary': {
                'models_tested': list(self.models.keys()),
                'tasks_tested': task_ids,
                'experiment_type': 're-arc'
            },
            'detailed_results': {}
        }
        
        for task_id in task_ids:
            print(f"\n{'='*80}")
            print(f"RE-ARC EXPERIMENT ON TASK: {task_id}")
            print(f"{'='*80}")
            
            task = self.load_arc_task(task_id)
            if not task:
                print(f"Task {task_id} not found!")
                continue
            
            augmented_examples = self.load_rearc_augmentations(task_id)
            
            task_results = {
                'task_id': task_id,
                'num_augmented_available': len(augmented_examples),
                'model_results': {}
            }
            
            for model_id, model in self.models.items():
                print(f"\n{'-'*60}")
                print(f"Testing Model: {model_id}")
                print(f"{'-'*60}")
                
                model_results = {}
                
                for num_aug in augmentation_sizes:
                    if num_aug > len(augmented_examples):
                        continue
                    
                    print(f"  Testing with {num_aug} augmented examples...")
                    
                    if num_aug == 0:
                        prompt = self.create_baseline_prompt(task, use_colors=True)
                    else:
                        prompt = self.create_augmented_prompt(task, augmented_examples, num_aug, use_colors=True)
                    
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
                    model_results[num_aug] = {
                        'accuracy': accuracy,
                        'correct_count': correct_count,
                        'total': num_candidates,
                        'responses_and_predictions': responses_and_predictions
                    }
                    print(f"    Accuracy: {accuracy:.2%}")
                
                task_results['model_results'][model_id] = model_results
                
                # W&B 로깅
                if self.use_wandb:
                    wandb.log({
                        'task_id': task_id,
                        'model_id': model_id,
                        'baseline_accuracy': model_results.get(0, {}).get('accuracy', 0),
                        'best_accuracy': max([r.get('accuracy', 0) for r in model_results.values()]) if model_results else 0
                    })
            
            all_results['detailed_results'][task_id] = task_results
        
        # 종합 요약 생성
        self.generate_summary(all_results)
        return all_results
    
    def generate_summary(self, results: Dict):
        """실험 결과 요약"""
        print("\n" + "="*80)
        print("RE-ARC EXPERIMENT SUMMARY")
        print("="*80)
        
        models = results['experiment_summary']['models_tested']
        tasks = results['experiment_summary']['tasks_tested']
        
        model_summary = {}
        for model_id in models:
            accuracies = []
            for task_id in tasks:
                if task_id in results['detailed_results']:
                    task_data = results['detailed_results'][task_id]
                    if model_id in task_data['model_results']:
                        model_data = task_data['model_results'][model_id]
                        accuracies.extend([r.get('accuracy', 0) for r in model_data.values()])
            
            model_summary[model_id] = {
                'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                'max_accuracy': max(accuracies) if accuracies else 0
            }
        
        print(f"\n{'Model':<20} {'Avg Accuracy':<15} {'Max Accuracy':<15}")
        print(f"{'='*20} {'='*15} {'='*15}")
        
        for model_id, summary in model_summary.items():
            print(f"{model_id:<20} {summary['avg_accuracy']:.1%}        {summary['max_accuracy']:.1%}")
        
        best_model = max(models, key=lambda m: model_summary[m]['max_accuracy'])
        print(f"\nBest Model: {best_model} ({model_summary[best_model]['max_accuracy']:.1%})")
        
        results['model_summary'] = model_summary
        results['best_model'] = best_model
    
    def save_results(self, results: Dict, output_file: str = "re_arc_results.json"):
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
RE-ARC Experiment

Usage: python re_arc_experiment.py [options]

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
    
    print(f"🚀 Starting RE-ARC Experiment")
    print(f"Selected models: {', '.join(models_config.keys())}")
    
    experiment = REARCExperiment(models_config, use_wandb=True)
    
    results = experiment.run_experiment(
        task_ids=["74dd1130", "1190e5a7", "150deff5", "178fcbfb"],
        augmentation_sizes=[0, 5, 10],
        num_candidates=3
    )
    
    experiment.save_results(results)
    
    if experiment.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()