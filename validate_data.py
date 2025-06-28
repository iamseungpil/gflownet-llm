import json
from pathlib import Path
import numpy as np

def validate_augmentation_data(task_id: str = "6150a2bd"):
    """RE-ARC와 H-ARC 데이터가 해당 문제에 대한 것인지 검증"""
    
    # 경로 설정
    arc_path = Path("../../arc-prize-2024/arc-agi_training_challenges.json")
    rearc_path = Path("../../re-arc") / f"{task_id}.json"
    harc_path = Path("../../h-arc") / f"{task_id}.json"
    
    print(f"Validating augmentation data for task: {task_id}")
    print("=" * 60)
    
    # 1. 원본 ARC 데이터 로드
    with open(arc_path, 'r') as f:
        arc_data = json.load(f)
    
    if task_id not in arc_data:
        print(f"Error: Task {task_id} not found in original ARC data!")
        return
    
    original_task = arc_data[task_id]
    print(f"\nOriginal ARC Task {task_id}:")
    print(f"- Training examples: {len(original_task['train'])}")
    print(f"- Test examples: {len(original_task['test'])}")
    
    # 원본 데이터의 그리드 크기 확인
    for i, example in enumerate(original_task['train']):
        input_shape = np.array(example['input']).shape
        output_shape = np.array(example['output']).shape
        print(f"  Train {i+1}: Input {input_shape} -> Output {output_shape}")
    
    # 2. RE-ARC 데이터 검증
    print(f"\n\nRE-ARC Data Validation:")
    print("-" * 40)
    
    if rearc_path.exists():
        with open(rearc_path, 'r') as f:
            rearc_data = json.load(f)
        
        print(f"RE-ARC file exists: {rearc_path}")
        print(f"Data type: {type(rearc_data)}")
        
        if isinstance(rearc_data, dict):
            print(f"Keys in RE-ARC data: {list(rearc_data.keys())[:10]}")
            
            # 데이터 구조 분석
            if 'train' in rearc_data:
                examples = rearc_data['train']
                print(f"Found 'train' key with {len(examples)} examples")
            elif isinstance(list(rearc_data.values())[0], dict) and 'input' in list(rearc_data.values())[0]:
                examples = list(rearc_data.values())
                print(f"Found {len(examples)} examples in dict format")
            else:
                examples = []
                print("Unknown RE-ARC format!")
        elif isinstance(rearc_data, list):
            examples = rearc_data
            print(f"Found {len(examples)} examples in list format")
        else:
            examples = []
            print(f"Unexpected data type: {type(rearc_data)}")
        
        # 예제 검증
        valid_count = 0
        for i, example in enumerate(examples[:5]):  # 처음 5개만 확인
            if isinstance(example, dict) and 'input' in example and 'output' in example:
                input_shape = np.array(example['input']).shape
                output_shape = np.array(example['output']).shape
                print(f"  Example {i+1}: Input {input_shape} -> Output {output_shape} ✓")
                valid_count += 1
            else:
                print(f"  Example {i+1}: Invalid format ✗")
        
        print(f"\nTotal valid RE-ARC examples: {valid_count}/{len(examples)}")
        
        # 변환 패턴 확인 (첫 번째 유효한 예제로)
        if valid_count > 0 and len(original_task['train']) > 0:
            print("\nChecking transformation consistency:")
            orig_input = np.array(original_task['train'][0]['input'])
            orig_output = np.array(original_task['train'][0]['output'])
            
            for i, example in enumerate(examples[:3]):
                if isinstance(example, dict) and 'input' in example and 'output' in example:
                    aug_input = np.array(example['input'])
                    aug_output = np.array(example['output'])
                    
                    # 간단한 변환 체크 (예: 대각선 뒤집기)
                    if aug_output.shape == aug_input.T.shape:
                        print(f"  Example {i+1}: Possible diagonal flip detected")
                    elif np.array_equal(aug_output, np.fliplr(aug_input)):
                        print(f"  Example {i+1}: Horizontal flip detected")
                    elif np.array_equal(aug_output, np.flipud(aug_input)):
                        print(f"  Example {i+1}: Vertical flip detected")
    else:
        print(f"RE-ARC file not found: {rearc_path}")
    
    # 3. H-ARC 데이터 검증
    print(f"\n\nH-ARC Data Validation:")
    print("-" * 40)
    
    if harc_path.exists():
        with open(harc_path, 'r') as f:
            harc_data = json.load(f)
        
        print(f"H-ARC file exists: {harc_path}")
        print(f"Data type: {type(harc_data)}")
        
        if isinstance(harc_data, dict):
            print(f"Keys in H-ARC data: {list(harc_data.keys())[:10]}")
            
            # 데이터 구조 분석
            if 'train' in harc_data:
                examples = harc_data['train']
                print(f"Found 'train' key with {len(examples)} examples")
            elif 'human_augmented' in harc_data:
                examples = harc_data['human_augmented']
                print(f"Found 'human_augmented' key with {len(examples)} examples")
            elif isinstance(list(harc_data.values())[0], dict) and 'input' in list(harc_data.values())[0]:
                examples = list(harc_data.values())
                print(f"Found {len(examples)} examples in dict format")
            else:
                examples = []
                print("Unknown H-ARC format!")
        elif isinstance(harc_data, list):
            examples = harc_data
            print(f"Found {len(examples)} examples in list format")
        else:
            examples = []
            print(f"Unexpected data type: {type(harc_data)}")
        
        # 예제 검증
        valid_count = 0
        for i, example in enumerate(examples[:5]):  # 처음 5개만 확인
            if isinstance(example, dict) and 'input' in example and 'output' in example:
                input_shape = np.array(example['input']).shape
                output_shape = np.array(example['output']).shape
                print(f"  Example {i+1}: Input {input_shape} -> Output {output_shape} ✓")
                valid_count += 1
            else:
                print(f"  Example {i+1}: Invalid format ✗")
        
        print(f"\nTotal valid H-ARC examples: {valid_count}/{len(examples)}")
    else:
        print(f"H-ARC file not found: {harc_path}")
    
    print("\n" + "=" * 60)
    print("Validation complete!")


if __name__ == "__main__":
    # Task 178 검증
    validate_augmentation_data("6150a2bd")
    
    # 추가 태스크 검증 (선택사항)
    # other_tasks = ["87a80de6", "9ddd00f0", "d43fd935"]
    # for task_id in other_tasks:
    #     print(f"\n\n{'='*80}\n")
    #     validate_augmentation_data(task_id)
