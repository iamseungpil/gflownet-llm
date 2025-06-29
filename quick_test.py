#!/usr/bin/env python3
"""
빠른 기능 테스트 (LLM 추론 없이 데이터 로딩만)
"""

import os
os.environ['WANDB_MODE'] = 'disabled'  # W&B 비활성화

def test_data_loading():
    """데이터 로딩 기능만 테스트"""
    print("="*60)
    print("Testing Data Loading (No LLM)")
    print("="*60)
    
    # RE-ARC 데이터 로딩 테스트
    print("\n1. Testing RE-ARC data loading...")
    from experiment_rearc import REARCExperiment
    
    # 모델 로딩 없이 인스턴스 생성 (데이터 로딩만 테스트)
    class REARCDataTest:
        def __init__(self):
            from pathlib import Path
            self.rearc_path = Path("data/re-arc/re_arc_extracted/re_arc/tasks")
        
        def load_rearc_augmentations(self, task_id):
            import json
            augmented_examples = []
            rearc_file = self.rearc_path / f"{task_id}.json"
            
            if rearc_file.exists():
                with open(rearc_file, 'r') as f:
                    rearc_data = json.load(f)
                    if isinstance(rearc_data, list):
                        augmented_examples = rearc_data
                        print(f"✅ Loaded {len(augmented_examples)} RE-ARC examples for task {task_id}")
            return augmented_examples
    
    rearc_test = REARCDataTest()
    rearc_examples = rearc_test.load_rearc_augmentations("6150a2bd")
    
    # H-ARC 데이터 로딩 테스트
    print("\n2. Testing H-ARC data loading...")
    
    class HARCDataTest:
        def __init__(self):
            from pathlib import Path
            self.harc_data_path = Path("data/h-arc/data/data.csv")
        
        def load_harc_action_traces(self, task_id):
            import pandas as pd
            if not self.harc_data_path.exists():
                print(f"❌ H-ARC data file not found: {self.harc_data_path}")
                return []
            
            df = pd.read_csv(self.harc_data_path)
            task_data = df[df['task_name'] == task_id]
            
            if len(task_data) == 0:
                print(f"❌ No H-ARC data found for task {task_id}")
                return []
            
            successful_attempts = task_data[task_data['solved'] == True]
            unique_attempts = successful_attempts[['hashed_id', 'attempt_number']].drop_duplicates()
            
            action_traces = []
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
                        'solved': action_row['solved']
                    }
                    actions.append(action_info)
                
                if actions:
                    action_traces.append(actions)
            
            print(f"✅ Loaded {len(action_traces)} successful action traces for task {task_id}")
            return action_traces
    
    harc_test = HARCDataTest()
    harc_traces = harc_test.load_harc_action_traces("6150a2bd.json")
    
    # ARC 태스크 로딩 테스트
    print("\n3. Testing ARC task loading...")
    from arc import train_problems, validation_problems
    
    all_problems = train_problems + validation_problems
    task = None
    for problem in all_problems:
        if problem.uid == "6150a2bd":
            task = problem
            break
    
    if task:
        print(f"✅ Found ARC task 6150a2bd with {len(task.train_pairs)} training examples")
        print(f"   Input shape: {task.train_pairs[0].x.shape}")
        print(f"   Output shape: {task.train_pairs[0].y.shape}")
    else:
        print("❌ ARC task 6150a2bd not found")
    
    # 프롬프트 생성 테스트
    print("\n4. Testing prompt generation...")
    
    if task and rearc_examples and harc_traces:
        # RE-ARC 프롬프트
        prompt_rearc = f"""Testing RE-ARC prompt with {len(rearc_examples)} examples...
Original training examples: {len(task.train_pairs)}
First augmented example input shape: {len(rearc_examples[0]['input'])}x{len(rearc_examples[0]['input'][0])}"""
        print(f"✅ RE-ARC prompt generation ready")
        
        # H-ARC 프롬프트
        first_trace_actions = [action['action'] for action in harc_traces[0][:5]]
        prompt_harc = f"""Testing H-ARC prompt with {len(harc_traces)} traces...
First trace has {len(harc_traces[0])} actions
First 5 actions: {first_trace_actions}"""
        print(f"✅ H-ARC prompt generation ready")
    
    print("\n" + "="*60)
    print("Data Loading Test Summary")
    print("="*60)
    
    success_count = 0
    if rearc_examples:
        print(f"✅ RE-ARC: {len(rearc_examples)} examples loaded")
        success_count += 1
    else:
        print("❌ RE-ARC: No examples loaded")
    
    if harc_traces:
        print(f"✅ H-ARC: {len(harc_traces)} traces loaded")
        success_count += 1
    else:
        print("❌ H-ARC: No traces loaded")
    
    if task:
        print(f"✅ ARC: Task loaded successfully")
        success_count += 1
    else:
        print("❌ ARC: Task loading failed")
    
    if success_count == 3:
        print("\n🎉 All data loading tests passed!")
        print("Ready to run full experiments!")
        return True
    else:
        print(f"\n⚠️  {success_count}/3 tests passed")
        return False

if __name__ == "__main__":
    success = test_data_loading()
    
    if success:
        print("\n📝 To run full experiments:")
        print("   python run_experiments.py")
        print("\n📝 To run individual experiments:")
        print("   python experiment_rearc.py")
        print("   python experiment_harc.py")