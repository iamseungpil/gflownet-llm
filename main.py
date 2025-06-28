import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Dict, List, Tuple
import re

class ARCLLMExperiment:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
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
        
    def load_arc_task(self, task_id: str) -> Dict:
        """원본 ARC 태스크 로드"""
        with open(self.arc_path, 'r') as f:
            data = json.load(f)
        return data.get(task_id, None)
    
    def load_rearc_augmentations(self, task_id: str) -> List[Dict]:
        """RE-ARC 증강 데이터 로드"""
        augmented_examples = []
        rearc_file = self.rearc_path / f"{task_id}.json"
        
        if rearc_file.exists():
            with open(rearc_file, 'r') as f:
                rearc_data = json.load(f)
                # RE-ARC 형식에서 증강된 예제 추출
                if 'augmented_examples' in rearc_data:
                    augmented_examples = rearc_data['augmented_examples'][:5]  # 최대 5개
        
        return augmented_examples
    
    def load_harc_augmentations(self, task_id: str) -> List[Dict]:
        """H-ARC 증강 데이터 로드"""
        augmented_examples = []
        harc_file = self.harc_path / f"{task_id}.json"
        
        if harc_file.exists():
            with open(harc_file, 'r') as f:
                harc_data = json.load(f)
                # H-ARC 형식에서 증강된 예제 추출
                if 'human_augmented' in harc_data:
                    augmented_examples = harc_data['human_augmented'][:5]  # 최대 5개
        
        return augmented_examples
    
    def grid_to_string(self, grid: List[List[int]]) -> str:
        """2D 그리드를 문자열로 변환"""
        return '\n'.join([' '.join(map(str, row)) for row in grid])
    
    def string_to_grid(self, grid_str: str) -> List[List[int]]:
        """문자열을 2D 그리드로 변환"""
        try:
            lines = grid_str.strip().split('\n')
            grid = []
            for line in lines:
                if line.strip():
                    row = [int(x) for x in line.split()]
                    grid.append(row)
            return grid
        except:
            return []
    
    def create_baseline_prompt(self, task: Dict) -> str:
        """기본 프롬프트 생성 (원본 데이터만 사용)"""
        prompt = """You are solving an ARC (Abstraction and Reasoning Corpus) task.
Find the pattern in the input-output examples and apply it to the test input.

Training Examples:
"""
        for i, example in enumerate(task['train']):
            prompt += f"\nExample {i+1}:\nInput:\n{self.grid_to_string(example['input'])}\n"
            prompt += f"Output:\n{self.grid_to_string(example['output'])}\n"
        
        prompt += f"\nTest Input:\n{self.grid_to_string(task['test'][0]['input'])}\n"
        prompt += "\nProvide only the output grid in the same format (numbers separated by spaces, rows on new lines):"
        
        return prompt
    
    def create_augmented_prompt(self, task: Dict, augmented_examples: List[Dict], 
                               augmentation_type: str) -> str:
        """증강 데이터를 포함한 프롬프트 생성"""
        prompt = f"""You are solving an ARC task with {augmentation_type} augmented examples.
Find the pattern and apply it to the test input.

Original Training Examples:
"""
        for i, example in enumerate(task['train']):
            prompt += f"\nExample {i+1}:\nInput:\n{self.grid_to_string(example['input'])}\n"
            prompt += f"Output:\n{self.grid_to_string(example['output'])}\n"
        
        if augmented_examples:
            prompt += f"\n{augmentation_type} Augmented Examples:"
            for i, example in enumerate(augmented_examples):
                prompt += f"\nAugmented Example {i+1}:\nInput:\n{self.grid_to_string(example['input'])}\n"
                prompt += f"Output:\n{self.grid_to_string(example['output'])}\n"
        
        prompt += f"\nTest Input:\n{self.grid_to_string(task['test'][0]['input'])}\n"
        prompt += "\nProvide only the output grid in the same format:"
        
        return prompt
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """LLM으로 응답 생성"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=2048).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                       skip_special_tokens=True)
        return response
    
    def extract_grid_from_response(self, response: str) -> List[List[int]]:
        """LLM 응답에서 그리드 추출"""
        # 숫자로만 이루어진 줄들을 찾기
        lines = response.strip().split('\n')
        grid_lines = []
        
        for line in lines:
            # 숫자와 공백만 포함된 줄 찾기
            if re.match(r'^[\d\s]+$', line.strip()) and line.strip():
                grid_lines.append(line.strip())
        
        # 연속된 그리드 라인들을 찾기
        if grid_lines:
            return self.string_to_grid('\n'.join(grid_lines))
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
    
    def run_experiment(self, task_id: str = "6150a2bd") -> Dict:
        """실험 실행 (Task 178: Diagonal Flip)"""
        print(f"\nRunning experiment on task: {task_id}")
        
        # 데이터 로드
        task = self.load_arc_task(task_id)
        if not task:
            print(f"Task {task_id} not found!")
            return {}
        
        rearc_examples = self.load_rearc_augmentations(task_id)
        harc_examples = self.load_harc_augmentations(task_id)
        
        results = {
            'task_id': task_id,
            'baseline': {'correct': False, 'response': None, 'predicted_grid': None},
            're-arc': {'correct': False, 'response': None, 'predicted_grid': None, 
                      'num_augmented': len(rearc_examples)},
            'h-arc': {'correct': False, 'response': None, 'predicted_grid': None,
                     'num_augmented': len(harc_examples)}
        }
        
        expected_output = task['test'][0]['output']
        
        # 1. Baseline 실험
        print("\n1. Testing Baseline (original data only)...")
        baseline_prompt = self.create_baseline_prompt(task)
        baseline_response = self.generate_response(baseline_prompt)
        baseline_grid = self.extract_grid_from_response(baseline_response)
        results['baseline']['response'] = baseline_response
        results['baseline']['predicted_grid'] = baseline_grid
        results['baseline']['correct'] = self.evaluate_solution(baseline_grid, expected_output)
        print(f"Baseline correct: {results['baseline']['correct']}")
        
        # 2. RE-ARC 실험
        print(f"\n2. Testing RE-ARC (with {len(rearc_examples)} augmented examples)...")
        rearc_prompt = self.create_augmented_prompt(task, rearc_examples, "RE-ARC")
        rearc_response = self.generate_response(rearc_prompt)
        rearc_grid = self.extract_grid_from_response(rearc_response)
        results['re-arc']['response'] = rearc_response
        results['re-arc']['predicted_grid'] = rearc_grid
        results['re-arc']['correct'] = self.evaluate_solution(rearc_grid, expected_output)
        print(f"RE-ARC correct: {results['re-arc']['correct']}")
        
        # 3. H-ARC 실험
        print(f"\n3. Testing H-ARC (with {len(harc_examples)} augmented examples)...")
        harc_prompt = self.create_augmented_prompt(task, harc_examples, "H-ARC")
        harc_response = self.generate_response(harc_prompt)
        harc_grid = self.extract_grid_from_response(harc_response)
        results['h-arc']['response'] = harc_response
        results['h-arc']['predicted_grid'] = harc_grid
        results['h-arc']['correct'] = self.evaluate_solution(harc_grid, expected_output)
        print(f"H-ARC correct: {results['h-arc']['correct']}")
        
        # 결과 요약
        print("\n=== Summary ===")
        print(f"Task ID: {task_id}")
        print(f"Baseline: {'✓' if results['baseline']['correct'] else '✗'}")
        print(f"RE-ARC: {'✓' if results['re-arc']['correct'] else '✗'} "
              f"(+{results['re-arc']['num_augmented']} examples)")
        print(f"H-ARC: {'✓' if results['h-arc']['correct'] else '✗'} "
              f"(+{results['h-arc']['num_augmented']} examples)")
        
        return results
    
    def save_results(self, results: Dict, output_file: str = "experiment_results.json"):
        """결과 저장"""
        output_path = Path(output_file)
        
        # 그리드를 JSON 직렬화 가능한 형태로 변환
        for method in ['baseline', 're-arc', 'h-arc']:
            if results[method]['predicted_grid'] is not None:
                results[method]['predicted_grid'] = results[method]['predicted_grid']
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """메인 실행 함수"""
    # 실험 초기화
    experiment = ARCLLMExperiment()
    
    # Task 178 (Diagonal Flip) 실험 실행
    results = experiment.run_experiment(task_id="6150a2bd")
    
    # 결과 저장
    experiment.save_results(results)
    
    # 추가 태스크 실험 (선택사항)
    # additional_tasks = ["87a80de6", "9ddd00f0", "d43fd935"]  # 논문의 다른 태스크들
    # for task_id in additional_tasks:
    #     results = experiment.run_experiment(task_id)
    #     experiment.save_results(results, f"results_{task_id}.json")


if __name__ == "__main__":
    main()
