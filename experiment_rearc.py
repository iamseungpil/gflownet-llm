import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import wandb
from tqdm import tqdm
from arc import train_problems, validation_problems

# Color mapping for ARC grids
COLOR_MAP = {
    0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
    5: "Gray", 6: "Pink", 7: "Orange", 8: "Purple", 9: "Brown"
}

REVERSE_COLOR_MAP = {v: k for k, v in COLOR_MAP.items()}

class REARCExperiment:
    """
    RE-ARC 실험: 증강된 input-output 그리드 쌍을 사용한 LLM 실험
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
        
        # RE-ARC 데이터 경로
        self.rearc_path = Path("data/re-arc/re_arc_extracted/re_arc/tasks")
        
        # W&B 초기화
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
            wandb.init(
                project="arc-llm-rearc-experiment",
                config={
                    "model": model_name,
                    "device": str(self.device),
                    "experiment_type": "re-arc_grid_augmentation"
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
                
                # RE-ARC 데이터는 리스트 형식
                if isinstance(rearc_data, list):
                    augmented_examples = rearc_data
                    print(f"Loaded {len(augmented_examples)} RE-ARC examples for task {task_id}")
                else:
                    print(f"Unexpected RE-ARC data format for task {task_id}")
        else:
            print(f"RE-ARC file not found for task {task_id}")
        
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
    
    def create_rearc_prompt(self, task: Dict, augmented_examples: List[Dict], 
                           num_augmented: int = 5, use_colors: bool = False) -> str:
        """RE-ARC 증강 데이터를 활용한 프롬프트 생성"""
        if use_colors:
            prompt = """You are solving an ARC (Abstraction and Reasoning Corpus) task.
The following examples show input-output transformations. Find the pattern and apply it to the test input.

Colors: Black(0), Blue(1), Red(2), Green(3), Yellow(4), Gray(5), Pink(6), Orange(7), Purple(8), Brown(9).

Original Training Examples:
"""
        else:
            prompt = """You are solving an ARC (Abstraction and Reasoning Corpus) task.
The following examples show input-output transformations. Find the pattern and apply it to the test input.

Original Training Examples:
"""
        
        # 원본 예제들
        for i, pair in enumerate(task.train_pairs):
            prompt += f"\nExample {i+1}:\nInput:\n{self.grid_to_string(pair.x.tolist(), use_colors)}\n"
            prompt += f"Output:\n{self.grid_to_string(pair.y.tolist(), use_colors)}\n"
        
        # RE-ARC 증강 예제들
        if augmented_examples and num_augmented > 0:
            selected_examples = augmented_examples[:num_augmented]
            prompt += f"\nAdditional Generated Examples (following the same pattern):"
            
            for i, example in enumerate(selected_examples):
                prompt += f"\n\nGenerated Example {i+1}:\nInput:\n{self.grid_to_string(example['input'], use_colors)}\n"
                prompt += f"Output:\n{self.grid_to_string(example['output'], use_colors)}\n"
        
        # 테스트 입력
        prompt += f"\n\nTest Input:\n{self.grid_to_string(task.test_pairs[0].x.tolist(), use_colors)}\n"
        
        if use_colors:
            prompt += "\nProvide only the output grid in the same format (colors separated by spaces, rows on new lines):"
        else:
            prompt += "\nProvide only the output grid in the same format (numbers separated by spaces, rows on new lines):"
        
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
    
    def run_rearc_experiment(self, task_ids: List[str] = None, 
                            augmented_sizes: List[int] = [0, 1, 3, 5, 10],
                            num_candidates: int = 5, use_colors: bool = False) -> Dict:
        """RE-ARC 실험 실행"""
        if task_ids is None:
            task_ids = ["6150a2bd", "87a80de6", "9ddd00f0", "d43fd935"]
        
        all_results = {}
        
        for task_id in task_ids:
            print(f"\n{'='*60}")
            print(f"Running RE-ARC experiment on task: {task_id}")
            print(f"{'='*60}")
            
            # 태스크와 증강 데이터 로드
            task = self.load_arc_task(task_id)
            if not task:
                print(f"Task {task_id} not found!")
                continue
            
            rearc_examples = self.load_rearc_augmentations(task_id)
            
            task_results = {
                'task_id': task_id,
                'num_rearc_examples': len(rearc_examples),
                'results_by_augmentation': {}
            }
            
            # 다양한 증강 데이터 개수로 실험
            for num_aug in augmented_sizes:
                if num_aug > len(rearc_examples):
                    continue
                
                print(f"\nTesting with {num_aug} RE-ARC augmented examples...")
                
                correct_count = 0
                for _ in range(num_candidates):
                    prompt = self.create_rearc_prompt(task, rearc_examples, num_aug, use_colors)
                    response = self.generate_response(prompt, temperature=0.3)
                    predicted_grid = self.extract_grid_from_response(response, use_colors)
                    
                    expected_output = task.test_pairs[0].y.tolist()
                    is_correct = self.evaluate_solution(predicted_grid, expected_output)
                    if is_correct:
                        correct_count += 1
                
                accuracy = correct_count / num_candidates
                task_results['results_by_augmentation'][num_aug] = {
                    'accuracy': accuracy,
                    'correct_count': correct_count,
                    'total_candidates': num_candidates
                }
                
                print(f"Accuracy with {num_aug} augmented examples: {accuracy:.2%}")
                
                if self.use_wandb:
                    wandb.log({
                        'task_id': task_id,
                        'num_augmented': num_aug,
                        'accuracy': accuracy,
                        'experiment': 're-arc'
                    })
            
            all_results[task_id] = task_results
        
        return all_results
    
    def save_results(self, results: Dict, output_file: str = "rearc_experiment_results.json"):
        """결과 저장"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main():
    """RE-ARC 실험 메인 함수"""
    experiment = REARCExperiment(use_wandb=True)
    
    # 실험할 태스크들
    task_ids = ["6150a2bd", "178fcbfb", "1190e5a7", "150deff5"]
    
    # 증강 데이터 개수 설정
    augmented_sizes = [0, 1, 2, 3, 5, 10, 20]
    
    # 실험 실행
    results = experiment.run_rearc_experiment(
        task_ids=task_ids,
        augmented_sizes=augmented_sizes,
        num_candidates=5,
        use_colors=False
    )
    
    # 결과 저장
    experiment.save_results(results)
    
    # 색상 이름을 사용한 추가 실험 (선택사항)
    print("\n\n=== Running experiment with color names ===")
    results_colors = experiment.run_rearc_experiment(
        task_ids=task_ids[:2],  # 처음 2개 태스크만
        augmented_sizes=augmented_sizes,
        num_candidates=5,
        use_colors=True
    )
    
    experiment.save_results(results_colors, "rearc_experiment_results_colors.json")
    
    if experiment.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
