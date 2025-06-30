"""
Multi-Model ARC Experiment
여러 LLM 모델(Llama 3.1, OpenAI O1, Gemma 7B, Qwen2.5-7B)로 ARC 실험을 수행합니다.
"""

import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
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
        """모델별 응답 생성 메서드"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """모델 이름 반환"""
        pass


class HuggingFaceModel(BaseARCModel):
    """Hugging Face 모델용 래퍼 (Llama, Gemma, Qwen)"""
    
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 모델별 특별 설정
        if "qwen" in model_name.lower():
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
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
        
        # API 키 설정
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        print(f"Initialized OpenAI {model_name}")
    
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        try:
            # o1 모델은 temperature를 지원하지 않음
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


class MultiModelARCExperiment:
    """여러 모델로 ARC 실험을 수행하는 클래스"""
    
    def __init__(self, models_config: Dict[str, Dict], use_wandb: bool = True):
        """
        models_config: {
            "model_id": {
                "type": "huggingface" or "openai",
                "model_name": "actual model name or path",
                "api_key": "for openai models" (optional)
            }
        }
        """
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
        
        # W&B 초기화
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
            wandb.init(
                project="arc-multi-model-comparison",
                config={
                    "models": list(models_config.keys()),
                    "experiment_type": "multi_model_comparison"
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
                
                # 데이터 검증
                valid_examples = []
                for example in augmented_examples:
                    if isinstance(example, dict) and 'input' in example and 'output' in example:
                        valid_examples.append(example)
                
                augmented_examples = valid_examples[:20]  # 최대 20개만 사용
        
        return augmented_examples
    
    def grid_to_string(self, grid: List[List[int]], use_colors: bool = False) -> str:
        """2D 그리드를 문자열로 변환"""
        if use_colors:
            rows = []
            for row in grid:
                color_row = [COLOR_MAP.get(int(cell), f"Unknown_{cell}") for cell in row]
                rows.append(" ".join(color_row))
            return "\n".join(rows)
        else:
            return '\n'.join([' '.join(map(str, row)) for row in grid])
    
    def string_to_grid(self, grid_str: str, from_colors: bool = False) -> List[List[int]]:
        """문자열을 2D 그리드로 변환"""
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
    
    def create_baseline_prompt(self, task: Dict, use_colors: bool = False) -> str:
        """기본 프롬프트 생성"""
        if use_colors:
            prompt = """You are solving an ARC (Abstraction and Reasoning Corpus) task.
Find the pattern in the input-output examples and apply it to the test input.

Each grid uses colors: Black(0), Blue(1), Red(2), Green(3), Yellow(4), Gray(5), Pink(6), Orange(7), Purple(8), Brown(9).

Training Examples:
"""
        else:
            prompt = """You are solving an ARC (Abstraction and Reasoning Corpus) task.
Find the pattern in the input-output examples and apply it to the test input.

Training Examples:
"""
        
        for i, pair in enumerate(task.train_pairs):
            prompt += f"\nExample {i+1}:\nInput:\n{self.grid_to_string(pair.x.tolist(), use_colors)}\n"
            prompt += f"Output:\n{self.grid_to_string(pair.y.tolist(), use_colors)}\n"
        
        prompt += f"\nTest Input:\n{self.grid_to_string(task.test_pairs[0].x.tolist(), use_colors)}\n"
        if use_colors:
            prompt += "\nProvide only the output grid in the same format (colors separated by spaces, rows on new lines):"
        else:
            prompt += "\nProvide only the output grid in the same format (numbers separated by spaces, rows on new lines):"
        
        return prompt
    
    def create_augmented_prompt(self, task: Dict, augmented_examples: List[Dict], 
                               num_augmented: int = 5, use_colors: bool = False) -> str:
        """증강 데이터를 포함한 프롬프트 생성"""
        if use_colors:
            prompt = """You are solving an ARC task with additional examples.
Find the pattern and apply it to the test input.

Colors: Black(0), Blue(1), Red(2), Green(3), Yellow(4), Gray(5), Pink(6), Orange(7), Purple(8), Brown(9).

Original Training Examples:
"""
        else:
            prompt = """You are solving an ARC task with additional examples.
Find the pattern and apply it to the test input.

Original Training Examples:
"""
        
        for i, pair in enumerate(task.train_pairs):
            prompt += f"\nExample {i+1}:\nInput:\n{self.grid_to_string(pair.x.tolist(), use_colors)}\n"
            prompt += f"Output:\n{self.grid_to_string(pair.y.tolist(), use_colors)}\n"
        
        # 증강 데이터 추가
        if augmented_examples and num_augmented > 0:
            selected_examples = augmented_examples[:num_augmented]
            prompt += f"\nAdditional Examples:"
            for i, example in enumerate(selected_examples):
                prompt += f"\nAugmented Example {i+1}:\nInput:\n{self.grid_to_string(example['input'], use_colors)}\n"
                prompt += f"Output:\n{self.grid_to_string(example['output'], use_colors)}\n"
        
        prompt += f"\nTest Input:\n{self.grid_to_string(task.test_pairs[0].x.tolist(), use_colors)}\n"
        if use_colors:
            prompt += "\nProvide only the output grid in the same format (colors separated by spaces, rows on new lines):"
        else:
            prompt += "\nProvide only the output grid in the same format (numbers separated by spaces, rows on new lines):"
        
        return prompt
    
    def extract_grid_from_response(self, response: str, from_colors: bool = False) -> List[List[int]]:
        """LLM 응답에서 그리드 추출"""
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
        """예측된 출력과 정답 비교"""
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
    
    def run_model_comparison(self, task_ids: List[str] = None,
                           augmentation_sizes: List[int] = [0, 5, 10],
                           num_candidates: int = 5,
                           use_colors: bool = False) -> Dict:
        """모든 모델에 대해 비교 실험 실행"""
        if task_ids is None:
            task_ids = ["6150a2bd", "178fcbfb", "1190e5a7", "150deff5"]
        
        all_results = {
            'task_results': {},
            'model_summary': {},
            'augmentation_summary': {}
        }
        
        # 각 태스크에 대해
        for task_id in task_ids:
            print(f"\n{'='*60}")
            print(f"Task: {task_id}")
            print(f"{'='*60}")
            
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
            
            # 각 모델에 대해
            for model_id, model in self.models.items():
                print(f"\n--- Testing with {model_id} ---")
                model_results = {
                    'baseline': {},
                    'augmented': {}
                }
                
                # 1. Baseline 실험 (증강 없음)
                print("Testing baseline (no augmentation)...")
                prompt = self.create_baseline_prompt(task, use_colors)
                
                correct_count = 0
                for _ in range(num_candidates):
                    response = model.generate_response(prompt, temperature=0.3)
                    predicted_grid = self.extract_grid_from_response(response, use_colors)
                    expected_output = task.test_pairs[0].y.tolist()
                    
                    if self.evaluate_solution(predicted_grid, expected_output):
                        correct_count += 1
                
                baseline_accuracy = correct_count / num_candidates
                model_results['baseline'] = {
                    'accuracy': baseline_accuracy,
                    'correct_count': correct_count,
                    'total': num_candidates
                }
                print(f"Baseline accuracy: {baseline_accuracy:.2%}")
                
                # 2. 증강 데이터 실험
                for num_aug in augmentation_sizes:
                    if num_aug > len(augmented_examples):
                        continue
                    
                    print(f"Testing with {num_aug} augmented examples...")
                    prompt = self.create_augmented_prompt(task, augmented_examples, num_aug, use_colors)
                    
                    correct_count = 0
                    for _ in range(num_candidates):
                        response = model.generate_response(prompt, temperature=0.3)
                        predicted_grid = self.extract_grid_from_response(response, use_colors)
                        
                        if self.evaluate_solution(predicted_grid, expected_output):
                            correct_count += 1
                    
                    aug_accuracy = correct_count / num_candidates
                    model_results['augmented'][num_aug] = {
                        'accuracy': aug_accuracy,
                        'correct_count': correct_count,
                        'total': num_candidates
                    }
                    print(f"Accuracy with {num_aug} augmented: {aug_accuracy:.2%}")
                
                task_results['model_results'][model_id] = model_results
                
                # W&B 로깅
                if self.use_wandb:
                    wandb.log({
                        'task_id': task_id,
                        'model_id': model_id,
                        'baseline_accuracy': model_results['baseline']['accuracy'],
                        **{f'aug_{n}_accuracy': model_results['augmented'].get(n, {}).get('accuracy', 0) 
                           for n in augmentation_sizes}
                    })
            
            all_results['task_results'][task_id] = task_results
        
        # 모델별 요약 통계
        self.compute_summary_statistics(all_results)
        
        return all_results
    
    def compute_summary_statistics(self, results: Dict):
        """결과 요약 통계 계산"""
        model_summary = {}
        
        for model_id in self.models.keys():
            model_accuracies = {
                'baseline': [],
                'augmented': {n: [] for n in [0, 5, 10]}
            }
            
            for task_id, task_results in results['task_results'].items():
                if model_id in task_results['model_results']:
                    model_data = task_results['model_results'][model_id]
                    model_accuracies['baseline'].append(model_data['baseline']['accuracy'])
                    
                    for num_aug, aug_data in model_data['augmented'].items():
                        if num_aug in model_accuracies['augmented']:
                            model_accuracies['augmented'][num_aug].append(aug_data['accuracy'])
            
            # 평균 계산
            model_summary[model_id] = {
                'baseline_avg': np.mean(model_accuracies['baseline']) if model_accuracies['baseline'] else 0,
                'augmented_avg': {
                    n: np.mean(accs) if accs else 0 
                    for n, accs in model_accuracies['augmented'].items()
                }
            }
        
        results['model_summary'] = model_summary
        
        # 출력
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        for model_id, summary in model_summary.items():
            print(f"\n{model_id}:")
            print(f"  Baseline: {summary['baseline_avg']:.2%}")
            for num_aug, acc in sorted(summary['augmented_avg'].items()):
                if acc > 0:
                    print(f"  With {num_aug} augmented: {acc:.2%}")
    
    def save_results(self, results: Dict, output_file: str = "multi_model_results.json"):
        """결과 저장"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def run_quick_test():
    """빠른 테스트를 위한 간단한 실행"""
    models_config = {
        "llama3.1-8b": {
            "type": "huggingface",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct"
        }
    }
    
    experiment = MultiModelARCExperiment(models_config, use_wandb=False)
    
    results = experiment.run_model_comparison(
        task_ids=["6150a2bd"],
        augmentation_sizes=[0, 5],
        num_candidates=2,
        use_colors=False
    )
    
    experiment.save_results(results, "quick_multi_model_test.json")
    print("Quick test completed!")


def main():
    """메인 실행 함수"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_test()
        return
    
    # 모델 설정
    models_config = {
        "llama3.1-8b": {
            "type": "huggingface",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct"
        },
        "openai-o1": {
            "type": "openai",
            "model_name": "o1-preview",
            "api_key": ""
        },
        "gemma-7b": {
            "type": "huggingface",
            "model_name": "google/gemma-7b-it"
        },
        "qwen2.5-7b": {
            "type": "huggingface",
            "model_name": "Qwen/Qwen2.5-7B-Instruct"
        }
    }
    
    # 실험 초기화
    experiment = MultiModelARCExperiment(models_config, use_wandb=True)
    
    # 실험할 태스크
    task_ids = ["74dd1130"]
    
    # 실험 실행
    results = experiment.run_model_comparison(
        task_ids=task_ids,
        augmentation_sizes=[0, 5, 10],
        num_candidates=5,
        use_colors=False
    )
    
    # 결과 저장
    experiment.save_results(results)
    
    if experiment.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()