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

# Color mapping for ARC grids (from the provided code)
COLOR_MAP = {
    0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
    5: "Gray", 6: "Pink", 7: "Orange", 8: "Purple", 9: "Brown"
}

REVERSE_COLOR_MAP = {v: k for k, v in COLOR_MAP.items()}

class ARCLLMExperiment:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", use_wandb=True):
        """
        ARC 문제를 LLM으로 해결하는 실험 클래스
        """
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # 데이터 경로 설정
        self.arc_path = Path("../../arc-prize-2024/arc-agi_training_challenges.json")
        self.rearc_path = Path("../../re-arc")
        self.harc_path = Path("../../h-arc")
        
        # W&B 초기화
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(
                project="arc-llm-augmentation",
                config={
                    "model": model_name,
                    "device": str(self.device)
                }
            )
    
    def load_arc_task(self, task_id: str) -> Dict:
        """원본 ARC 태스크 로드"""
        with open(self.arc_path, 'r') as f:
            data = json.load(f)
        return data.get(task_id, None)
    
    def load_rearc_augmentations(self, task_id: str) -> List[Dict]:
        """RE-ARC 증강 데이터 로드 및 검증"""
        augmented_examples = []
        rearc_file = self.rearc_path / f"{task_id}.json"
        
        if rearc_file.exists():
            with open(rearc_file, 'r') as f:
                rearc_data = json.load(f)
                
                # 데이터 형식 확인
                print(f"\nRE-ARC data structure for {task_id}:")
                print(f"Keys: {list(rearc_data.keys())[:5]}...")
                
                # RE-ARC 형식에 따라 데이터 추출
                if isinstance(rearc_data, list):
                    # 리스트 형식인 경우
                    augmented_examples = rearc_data
                elif 'train' in rearc_data:
                    # 기존 ARC 형식과 유사한 경우
                    augmented_examples = rearc_data.get('train', [])
                elif 'augmented_examples' in rearc_data:
                    augmented_examples = rearc_data['augmented_examples']
                else:
                    # 다른 형식인 경우 첫 번째 키의 값 사용
                    first_key = list(rearc_data.keys())[0]
                    if isinstance(rearc_data[first_key], dict) and 'input' in rearc_data[first_key]:
                        augmented_examples = [rearc_data[key] for key in list(rearc_data.keys())]
                
                # 데이터 검증
                valid_examples = []
                for i, example in enumerate(augmented_examples):
                    if isinstance(example, dict) and 'input' in example and 'output' in example:
                        valid_examples.append(example)
                        if i < 2:  # 처음 2개 예제 확인
                            print(f"RE-ARC Example {i+1} - Input shape: {np.array(example['input']).shape}")
                
                augmented_examples = valid_examples
                print(f"Loaded {len(augmented_examples)} valid RE-ARC examples")
        
        return augmented_examples
    
    def load_harc_augmentations(self, task_id: str) -> List[Dict]:
        """H-ARC 증강 데이터 로드 및 검증"""
        augmented_examples = []
        harc_file = self.harc_path / f"{task_id}.json"
        
        if harc_file.exists():
            with open(harc_file, 'r') as f:
                harc_data = json.load(f)
                
                # 데이터 형식 확인
                print(f"\nH-ARC data structure for {task_id}:")
                print(f"Keys: {list(harc_data.keys())[:5]}...")
                
                # H-ARC 형식에 따라 데이터 추출
                if isinstance(harc_data, list):
                    augmented_examples = harc_data
                elif 'train' in harc_data:
                    augmented_examples = harc_data.get('train', [])
                elif 'human_augmented' in harc_data:
                    augmented_examples = harc_data['human_augmented']
                else:
                    # 다른 형식인 경우
                    first_key = list(harc_data.keys())[0]
                    if isinstance(harc_data[first_key], dict) and 'input' in harc_data[first_key]:
                        augmented_examples = [harc_data[key] for key in list(harc_data.keys())]
                
                # 데이터 검증
                valid_examples = []
                for i, example in enumerate(augmented_examples):
                    if isinstance(example, dict) and 'input' in example and 'output' in example:
                        valid_examples.append(example)
                        if i < 2:  # 처음 2개 예제 확인
                            print(f"H-ARC Example {i+1} - Input shape: {np.array(example['input']).shape}")
                
                augmented_examples = valid_examples
                print(f"Loaded {len(augmented_examples)} valid H-ARC examples")
        
        return augmented_examples
    
    def grid_to_string(self, grid: List[List[int]], use_colors: bool = False) -> str:
        """2D 그리드를 문자열로 변환 (옵션: 색상 이름 사용)"""
        if use_colors:
            rows = []
            for row in grid:
                color_row = [COLOR_MAP.get(int(cell), f"Unknown_{cell}") for cell in row]
                rows.append(" ".join(color_row))
            return "\n".join(rows)
        else:
            return '\n'.join([' '.join(map(str, row)) for row in grid])
    
    def string_to_grid(self, grid_str: str, from_colors: bool = False) -> List[List[int]]:
        """문자열을 2D 그리드로 변환 (옵션: 색상 이름에서 변환)"""
        try:
            lines = grid_str.strip().split('\n')
            grid = []
            for line in lines:
                if line.strip():
                    if from_colors:
                        # 색상 이름을 숫자로 변환
                        colors = line.split()
                        row = [REVERSE_COLOR_MAP.get(color, -1) for color in colors]
                    else:
                        row = [int(x) for x in line.split()]
                    grid.append(row)
            return grid
        except:
            return []
    
    def create_baseline_prompt(self, task: Dict, num_examples: Optional[int] = None, use_colors: bool = False) -> str:
        """기본 프롬프트 생성 (원본 데이터만 사용)"""
        if use_colors:
            prompt = """You are solving an ARC (Abstraction and Reasoning Corpus) task.
Find the pattern in the input-output examples and apply it to the test input.

Each grid is represented as a 2D array where each cell is represented by a color.
The colors are: Black(0), Blue(1), Red(2), Green(3), Yellow(4), Gray(5), Pink(6), Orange(7), Purple(8), Brown(9).

Training Examples:
"""
        else:
            prompt = """You are solving an ARC (Abstraction and Reasoning Corpus) task.
Find the pattern in the input-output examples and apply it to the test input.

Training Examples:
"""
        train_examples = task['train'][:num_examples] if num_examples else task['train']
        
        for i, example in enumerate(train_examples):
            prompt += f"\nExample {i+1}:\nInput:\n{self.grid_to_string(example['input'], use_colors)}\n"
            prompt += f"Output:\n{self.grid_to_string(example['output'], use_colors)}\n"
        
        prompt += f"\nTest Input:\n{self.grid_to_string(task['test'][0]['input'], use_colors)}\n"
        if use_colors:
            prompt += "\nProvide only the output grid in the same format (colors separated by spaces, rows on new lines):"
        else:
            prompt += "\nProvide only the output grid in the same format (numbers separated by spaces, rows on new lines):"
        
        return prompt
    
    def create_augmented_prompt(self, task: Dict, augmented_examples: List[Dict], 
                               augmentation_type: str, num_augmented: int, use_colors: bool = False) -> str:
        """증강 데이터를 포함한 프롬프트 생성 (증강 데이터 개수 제어)"""
        if use_colors:
            prompt = f"""You are solving an ARC task with {augmentation_type} augmented examples.
Find the pattern and apply it to the test input.

Each grid is represented using colors: Black(0), Blue(1), Red(2), Green(3), Yellow(4), Gray(5), Pink(6), Orange(7), Purple(8), Brown(9).

Original Training Examples:
"""
        else:
            prompt = f"""You are solving an ARC task with {augmentation_type} augmented examples.
Find the pattern and apply it to the test input.

Original Training Examples:
"""
        for i, example in enumerate(task['train']):
            prompt += f"\nExample {i+1}:\nInput:\n{self.grid_to_string(example['input'], use_colors)}\n"
            prompt += f"Output:\n{self.grid_to_string(example['output'], use_colors)}\n"
        
        # 지정된 개수만큼 증강 데이터 추가
        if augmented_examples and num_augmented > 0:
            selected_examples = augmented_examples[:num_augmented]
            prompt += f"\n{augmentation_type} Augmented Examples ({len(selected_examples)} examples):"
            for i, example in enumerate(selected_examples):
                prompt += f"\nAugmented Example {i+1}:\nInput:\n{self.grid_to_string(example['input'], use_colors)}\n"
                prompt += f"Output:\n{self.grid_to_string(example['output'], use_colors)}\n"
        
        prompt += f"\nTest Input:\n{self.grid_to_string(task['test'][0]['input'], use_colors)}\n"
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
        """LLM 응답에서 그리드 추출"""
        lines = response.strip().split('\n')
        grid_lines = []
        
        if from_colors:
            # 색상 이름을 찾기
            color_pattern = '|'.join(COLOR_MAP.values())
            for line in lines:
                if re.search(color_pattern, line, re.IGNORECASE):
                    grid_lines.append(line.strip())
        else:
            # 숫자로만 이루어진 줄 찾기
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
    
    def evaluate_with_candidates(self, task: Dict, prompt: str, 
                                num_candidates: int = 5, use_colors: bool = False) -> Tuple[float, List[bool]]:
        """여러 candidate를 생성하여 정확도 측정"""
        expected_output = task['test'][0]['output']
        correct_predictions = []
        
        for _ in range(num_candidates):
            response = self.generate_response(prompt, temperature=0.3)  # 약간의 다양성
            predicted_grid = self.extract_grid_from_response(response, from_colors=use_colors)
            is_correct = self.evaluate_solution(predicted_grid, expected_output)
            correct_predictions.append(is_correct)
        
        accuracy = sum(correct_predictions) / len(correct_predictions)
        return accuracy, correct_predictions
    
    def run_data_scaling_experiment(self, task_id: str = "6150a2bd", 
                                   num_candidates: int = 5, use_colors: bool = False) -> Dict:
        """데이터 양 증가에 따른 정확도 변화 추적"""
        print(f"\nRunning data scaling experiment on task: {task_id}")
        print(f"Generating {num_candidates} candidates per configuration")
        print(f"Using colors: {use_colors}")
        
        # 데이터 로드
        task = self.load_arc_task(task_id)
        if not task:
            print(f"Task {task_id} not found!")
            return {}
        
        rearc_examples = self.load_rearc_augmentations(task_id)
        harc_examples = self.load_harc_augmentations(task_id)
        
        results = {
            'task_id': task_id,
            'num_candidates': num_candidates,
            'use_colors': use_colors,
            'baseline': {'accuracy_by_data_size': {}},
            're-arc': {'accuracy_by_data_size': {}, 'max_examples': len(rearc_examples)},
            'h-arc': {'accuracy_by_data_size': {}, 'max_examples': len(harc_examples)}
        }
        
        # 1. Baseline - 원본 데이터 개수 변화
        print("\n=== Baseline Experiment ===")
        baseline_sizes = range(1, len(task['train']) + 1)
        baseline_accuracies = []
        
        for num_examples in baseline_sizes:
            print(f"\nTesting with {num_examples} original examples...")
            prompt = self.create_baseline_prompt(task, num_examples, use_colors)
            accuracy, predictions = self.evaluate_with_candidates(task, prompt, num_candidates, use_colors)
            
            results['baseline']['accuracy_by_data_size'][num_examples] = {
                'accuracy': accuracy,
                'predictions': predictions
            }
            baseline_accuracies.append(accuracy)
            
            print(f"Accuracy: {accuracy:.2%}")
            
            if self.use_wandb:
                wandb.log({
                    'baseline_accuracy': accuracy,
                    'baseline_num_examples': num_examples
                })
        
        # 정확도 변화량 계산
        if len(baseline_accuracies) > 1:
            baseline_deltas = [baseline_accuracies[i] - baseline_accuracies[i-1] 
                              for i in range(1, len(baseline_accuracies))]
            results['baseline']['accuracy_deltas'] = baseline_deltas
        
        # 2. RE-ARC 실험
        if rearc_examples:
            print("\n=== RE-ARC Experiment ===")
            rearc_accuracies = []
            max_augmented = min(10, len(rearc_examples))  # 최대 10개까지
            
            for num_augmented in range(0, max_augmented + 1):
                print(f"\nTesting with {len(task['train'])} original + {num_augmented} RE-ARC examples...")
                prompt = self.create_augmented_prompt(task, rearc_examples, "RE-ARC", num_augmented, use_colors)
                accuracy, predictions = self.evaluate_with_candidates(task, prompt, num_candidates, use_colors)
                
                results['re-arc']['accuracy_by_data_size'][num_augmented] = {
                    'accuracy': accuracy,
                    'predictions': predictions
                }
                rearc_accuracies.append(accuracy)
                
                print(f"Accuracy: {accuracy:.2%}")
                
                if self.use_wandb:
                    wandb.log({
                        're-arc_accuracy': accuracy,
                        're-arc_num_augmented': num_augmented,
                        're-arc_total_examples': len(task['train']) + num_augmented
                    })
            
            # 정확도 변화량 계산
            if len(rearc_accuracies) > 1:
                rearc_deltas = [rearc_accuracies[i] - rearc_accuracies[i-1] 
                               for i in range(1, len(rearc_accuracies))]
                results['re-arc']['accuracy_deltas'] = rearc_deltas
        
        # 3. H-ARC 실험
        if harc_examples:
            print("\n=== H-ARC Experiment ===")
            harc_accuracies = []
            max_augmented = min(10, len(harc_examples))  # 최대 10개까지
            
            for num_augmented in range(0, max_augmented + 1):
                print(f"\nTesting with {len(task['train'])} original + {num_augmented} H-ARC examples...")
                prompt = self.create_augmented_prompt(task, harc_examples, "H-ARC", num_augmented, use_colors)
                accuracy, predictions = self.evaluate_with_candidates(task, prompt, num_candidates, use_colors)
                
                results['h-arc']['accuracy_by_data_size'][num_augmented] = {
                    'accuracy': accuracy,
                    'predictions': predictions
                }
                harc_accuracies.append(accuracy)
                
                print(f"Accuracy: {accuracy:.2%}")
                
                if self.use_wandb:
                    wandb.log({
                        'h-arc_accuracy': accuracy,
                        'h-arc_num_augmented': num_augmented,
                        'h-arc_total_examples': len(task['train']) + num_augmented
                    })
            
            # 정확도 변화량 계산
            if len(harc_accuracies) > 1:
                harc_deltas = [harc_accuracies[i] - harc_accuracies[i-1] 
                              for i in range(1, len(harc_accuracies))]
                results['h-arc']['accuracy_deltas'] = harc_deltas
        
        # 결과 요약 출력
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict):
        """결과 요약 출력"""
        print("\n=== SUMMARY ===")
        print(f"Task ID: {results['task_id']}")
        print(f"Candidates per configuration: {results['num_candidates']}")
        print(f"Using colors: {results.get('use_colors', False)}")
        
        # Baseline 요약
        print("\nBaseline Results:")
        for num_ex, data in results['baseline']['accuracy_by_data_size'].items():
            print(f"  {num_ex} examples: {data['accuracy']:.2%}")
        
        # RE-ARC 요약
        if results['re-arc']['accuracy_by_data_size']:
            print("\nRE-ARC Results:")
            for num_aug, data in results['re-arc']['accuracy_by_data_size'].items():
                print(f"  +{num_aug} augmented: {data['accuracy']:.2%}")
        
        # H-ARC 요약
        if results['h-arc']['accuracy_by_data_size']:
            print("\nH-ARC Results:")
            for num_aug, data in results['h-arc']['accuracy_by_data_size'].items():
                print(f"  +{num_aug} augmented: {data['accuracy']:.2%}")
        
        # 정확도 변화량 출력
        if 'accuracy_deltas' in results['baseline']:
            print("\nAccuracy Changes (Δ):")
            print(f"  Baseline: {[f'{d:+.2%}' for d in results['baseline']['accuracy_deltas']]}")
        if 'accuracy_deltas' in results['re-arc']:
            print(f"  RE-ARC: {[f'{d:+.2%}' for d in results['re-arc']['accuracy_deltas']]}")
        if 'accuracy_deltas' in results['h-arc']:
            print(f"  H-ARC: {[f'{d:+.2%}' for d in results['h-arc']['accuracy_deltas']]}")
    
    def save_results(self, results: Dict, output_file: str = "scaling_experiment_results.json"):
        """결과 저장"""
        output_path = Path(output_file)
        
        # predictions를 제거하고 저장 (파일 크기 줄이기)
        save_results = json.loads(json.dumps(results))  # deep copy
        for method in ['baseline', 're-arc', 'h-arc']:
            if method in save_results and 'accuracy_by_data_size' in save_results[method]:
                for key in save_results[method]['accuracy_by_data_size']:
                    if 'predictions' in save_results[method]['accuracy_by_data_size'][key]:
                        del save_results[method]['accuracy_by_data_size'][key]['predictions']
        
        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """메인 실행 함수"""
    # W&B 로그인 (필요한 경우)
    # wandb.login()
    
    # 실험 초기화
    experiment = ARCLLMExperiment(use_wandb=True)
    
    # 데이터 스케일링 실험 실행 (숫자 형식)
    print("\n=== Experiment with numeric grids ===")
    results_numeric = experiment.run_data_scaling_experiment(
        task_id="6150a2bd",  # Task 178
        num_candidates=5,    # 각 설정당 5개 candidate 생성
        use_colors=False     # 숫자 형식 사용
    )
    
    # 결과 저장
    experiment.save_results(results_numeric, "scaling_experiment_results_numeric.json")
    
    # 색상 형식으로 추가 실험 (선택사항)
    print("\n\n=== Experiment with color grids ===")
    results_colors = experiment.run_data_scaling_experiment(
        task_id="6150a2bd",
        num_candidates=5,
        use_colors=True      # 색상 형식 사용
    )
    
    # 결과 저장
    experiment.save_results(results_colors, "scaling_experiment_results_colors.json")
    
    # W&B 종료
    if experiment.use_wandb:
        wandb.finish()
    
    # 추가 태스크 실험 (선택사항)
    # additional_tasks = ["87a80de6", "9ddd00f0", "d43fd935"]
    # for task_id in additional_tasks:
    #     experiment = ARCLLMExperiment(use_wandb=True)
    #     results = experiment.run_data_scaling_experiment(task_id, use_colors=False)
    #     experiment.save_results(results, f"scaling_results_{task_id}.json")
    #     wandb.finish()


if __name__ == "__main__":
    main()
