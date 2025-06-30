import json
import csv
import os
from typing import List, Dict, Tuple, Any

def parse_trajectory(trajectory_data: List[Dict]) -> List[Dict]:
    """Trajectory 데이터를 파싱하여 action과 grid 상태를 추출"""
    parsed_actions = []
    
    for i, action in enumerate(trajectory_data):
        # Skip selection actions as they don't change the grid
        if action['category'] == 'Selection':
            continue
            
        # Skip the final submit action
        if action['operation'] == 'Submit':
            continue
            
        # Extract the action info
        action_info = {
            'index': i,
            'category': action['category'],
            'operation': action['operation'],
            'grid_before': action['grid'],
            'object_after': action['object']
        }
        
        # Add specific parameters for different operations
        if 'direction' in action:
            action_info['direction'] = action['direction']
        if 'color' in action:
            action_info['color'] = action['color']
        if 'position' in action:
            action_info['position'] = action['position']
            
        parsed_actions.append(action_info)
    
    return parsed_actions

def object_to_grid(objects: List[Dict], grid_size: Tuple[int, int] = (3, 3)) -> List[List[int]]:
    """Object 리스트를 grid 형태로 변환"""
    grid = [[0 for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    
    for obj in objects:
        x, y = obj['x'], obj['y']
        color = obj['color']
        if 0 <= y < grid_size[0] and 0 <= x < grid_size[1]:
            grid[y][x] = color
    
    return grid

def format_grid(grid: List[List[int]]) -> str:
    """Grid를 보기 좋은 문자열로 변환"""
    return '\n'.join([' '.join(map(str, row)) for row in grid])

def format_action(action: Dict) -> str:
    """Action을 문자열로 포맷팅"""
    action_str = f"{action['operation']}"
    
    if 'direction' in action:
        action_str += f"({action['direction']})"
    elif 'color' in action and action['operation'] == 'Paint':
        action_str += f"(color={action['color']})"
    elif 'position' in action and action['operation'] == 'SelectCell':
        action_str += f"(x={action['position']['x']}, y={action['position']['y']})"
    
    return action_str

def generate_sft_data(trajectory_data: List[Dict]) -> List[Dict]:
    """SFT를 위한 학습 데이터 생성"""
    parsed_actions = parse_trajectory(trajectory_data)
    sft_data = []
    
    # Get initial grid from the first selection
    initial_grid = None
    for action in trajectory_data:
        if action['category'] == 'Selection' and action['operation'] == 'SelectGrid':
            initial_grid = object_to_grid(action['object'])
            break
    
    if initial_grid is None:
        print("Warning: Could not find initial grid")
        return []
    
    # Build grid states for each action
    grid_states = [initial_grid]
    for action in parsed_actions:
        grid_after = object_to_grid(action['object_after'])
        grid_states.append(grid_after)
    
    # Generate training examples for different sequence lengths
    for seq_len in range(1, len(parsed_actions) + 1):
        for start_idx in range(len(parsed_actions) - seq_len + 1):
            end_idx = start_idx + seq_len
            
            # Get the sequence of actions
            action_sequence = parsed_actions[start_idx:end_idx]
            
            # Get before and after grids
            grid_before = grid_states[start_idx]
            grid_after = grid_states[end_idx]
            
            # Format actions
            actions_str = ' -> '.join([format_action(a) for a in action_sequence])
            
            # Create training example
            example = {
                'input': f"Initial grid:\n{format_grid(grid_before)}\n\nFinal grid:\n{format_grid(grid_after)}\n\nWhat action(s) transformed the initial grid to the final grid?",
                'output': actions_str,
                'sequence_length': seq_len,
                'start_idx': start_idx,
                'grid_before': grid_before,
                'grid_after': grid_after
            }
            
            sft_data.append(example)
    
    return sft_data

def load_arc_trajectory(csv_path: str, problem_id: str) -> List[Dict]:
    """ARCTraj.csv에서 특정 문제의 trajectory 로드"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('problem_id') == problem_id or row.get('id') == problem_id:
                    # Assuming trajectory is stored as JSON string in a column
                    # You may need to adjust this based on actual CSV structure
                    if 'trajectory' in row:
                        return json.loads(row['trajectory'])
                    elif 'data' in row:
                        return json.loads(row['data'])
        print(f"Problem {problem_id} not found in {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading trajectory from CSV: {e}")
        return None

def main():
    # Load trajectory data
    # First try to load from CSV
    csv_path = r"../data/o2arc/ARCTraj.csv"
    problem_id = "74dd1130"
    
    trajectory_data = None
    
    # Try loading from CSV
    if os.path.exists(csv_path):
        trajectory_data = load_arc_trajectory(csv_path, problem_id)
    
    # If not found in CSV, use the data from paste.txt
    if trajectory_data is None:
        print(f"Using trajectory data from paste.txt")
        # This is the data from paste.txt
        trajectory_data = json.loads('''[{"category":"Selection","operation":"SelectGrid","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":0,"x":0,"color":9},{"y":0,"x":1,"color":3},{"y":0,"x":2,"color":4},{"y":1,"x":0,"color":9},{"y":1,"x":1,"color":4},{"y":1,"x":2,"color":4},{"y":2,"x":0,"color":9},{"y":2,"x":1,"color":3},{"y":2,"x":2,"color":4}],"overlapped":true,"timestamp":"2024-02-15T02:25:21.923Z"},{"category":"O2","operation":"Flip","direction":"vertical","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":2,"x":0,"color":9},{"y":2,"x":1,"color":3},{"y":2,"x":2,"color":4},{"y":1,"x":0,"color":9},{"y":1,"x":1,"color":4},{"y":1,"x":2,"color":4},{"y":0,"x":0,"color":9},{"y":0,"x":1,"color":3},{"y":0,"x":2,"color":4}],"overlapped":false,"timestamp":"2024-02-15T02:25:22.521Z"},{"category":"O2","operation":"Flip","direction":"horizontal","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":2,"x":2,"color":9},{"y":2,"x":1,"color":3},{"y":2,"x":0,"color":4},{"y":1,"x":2,"color":9},{"y":1,"x":1,"color":4},{"y":1,"x":0,"color":4},{"y":0,"x":2,"color":9},{"y":0,"x":1,"color":3},{"y":0,"x":0,"color":4}],"overlapped":false,"timestamp":"2024-02-15T02:25:22.769Z"},{"category":"O2","operation":"Flip","direction":"vertical","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":0,"x":2,"color":9},{"y":0,"x":1,"color":3},{"y":0,"x":0,"color":4},{"y":1,"x":2,"color":9},{"y":1,"x":1,"color":4},{"y":1,"x":0,"color":4},{"y":2,"x":2,"color":9},{"y":2,"x":1,"color":3},{"y":2,"x":0,"color":4}],"overlapped":false,"timestamp":"2024-02-15T02:25:23.953Z"},{"category":"O2","operation":"Flip","direction":"vertical","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":2,"x":2,"color":9},{"y":2,"x":1,"color":3},{"y":2,"x":0,"color":4},{"y":1,"x":2,"color":9},{"y":1,"x":1,"color":4},{"y":1,"x":0,"color":4},{"y":0,"x":2,"color":9},{"y":0,"x":1,"color":3},{"y":0,"x":0,"color":4}],"overlapped":false,"timestamp":"2024-02-15T02:25:24.496Z"},{"category":"Selection","operation":"SelectCell","position":{"x":1,"y":1},"grid":[[4,3,9],[4,4,9],[4,3,9]],"object":[],"overlapped":true,"timestamp":"2024-02-15T02:25:26.493Z"},{"category":"Selection","operation":"SelectGrid","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":0,"x":0,"color":4},{"y":0,"x":1,"color":3},{"y":0,"x":2,"color":9},{"y":1,"x":0,"color":4},{"y":1,"x":1,"color":4},{"y":1,"x":2,"color":9},{"y":2,"x":0,"color":4},{"y":2,"x":1,"color":3},{"y":2,"x":2,"color":9}],"overlapped":true,"timestamp":"2024-02-15T02:25:27.184Z"},{"category":"O2","operation":"Flip","direction":"vertical","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":2,"x":0,"color":4},{"y":2,"x":1,"color":3},{"y":2,"x":2,"color":9},{"y":1,"x":0,"color":4},{"y":1,"x":1,"color":4},{"y":1,"x":2,"color":9},{"y":0,"x":0,"color":4},{"y":0,"x":1,"color":3},{"y":0,"x":2,"color":9}],"overlapped":false,"timestamp":"2024-02-15T02:25:28.010Z"},{"category":"O2","operation":"Flip","direction":"horizontal","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":2,"x":2,"color":4},{"y":2,"x":1,"color":3},{"y":2,"x":0,"color":9},{"y":1,"x":2,"color":4},{"y":1,"x":1,"color":4},{"y":1,"x":0,"color":9},{"y":0,"x":2,"color":4},{"y":0,"x":1,"color":3},{"y":0,"x":0,"color":9}],"overlapped":false,"timestamp":"2024-02-15T02:25:28.681Z"},{"category":"Selection","operation":"SelectCell","position":{"x":0,"y":1},"grid":[[9,3,4],[9,4,4],[9,3,4]],"object":[],"overlapped":true,"timestamp":"2024-02-15T02:25:36.689Z"},{"category":"Coloring","operation":"Paint","color":1,"grid":[[9,3,4],[9,4,4],[9,3,4]],"object":[],"overlapped":false,"timestamp":"2024-02-15T02:25:37.588Z"},{"category":"Selection","operation":"SelectGrid","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":0,"x":0,"color":9},{"y":0,"x":1,"color":3},{"y":0,"x":2,"color":4},{"y":1,"x":0,"color":9},{"y":1,"x":1,"color":4},{"y":1,"x":2,"color":4},{"y":2,"x":0,"color":9},{"y":2,"x":1,"color":3},{"y":2,"x":2,"color":4}],"overlapped":true,"timestamp":"2024-02-15T02:25:39.972Z"},{"category":"Selection","operation":"SelectCell","position":{"x":0,"y":2},"grid":[[9,3,4],[9,4,4],[9,3,4]],"object":[],"overlapped":true,"timestamp":"2024-02-15T02:25:40.418Z"},{"category":"Coloring","operation":"Paint","color":1,"grid":[[9,3,4],[9,4,4],[9,3,4]],"object":[],"overlapped":false,"timestamp":"2024-02-15T02:25:41.372Z"},{"category":"Selection","operation":"SelectCell","position":{"x":0,"y":2},"grid":[[9,3,4],[9,4,4],[0,3,4]],"object":[{"x":0,"y":2,"color":9}],"overlapped":true,"timestamp":"2024-02-15T02:25:41.937Z"},{"category":"Coloring","operation":"Paint","color":1,"grid":[[9,3,4],[9,4,4],[0,3,4]],"object":[{"x":0,"y":2,"color":1}],"overlapped":false,"timestamp":"2024-02-15T02:25:42.516Z"},{"category":"Selection","operation":"SelectGrid","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":0,"x":0,"color":9},{"y":0,"x":1,"color":3},{"y":0,"x":2,"color":4},{"y":1,"x":0,"color":9},{"y":1,"x":1,"color":4},{"y":1,"x":2,"color":4},{"y":2,"x":0,"color":1},{"y":2,"x":1,"color":3},{"y":2,"x":2,"color":4}],"overlapped":true,"timestamp":"2024-02-15T02:25:43.259Z"},{"category":"O2","operation":"Flip","direction":"vertical","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":2,"x":0,"color":9},{"y":2,"x":1,"color":3},{"y":2,"x":2,"color":4},{"y":1,"x":0,"color":9},{"y":1,"x":1,"color":4},{"y":1,"x":2,"color":4},{"y":0,"x":0,"color":1},{"y":0,"x":1,"color":3},{"y":0,"x":2,"color":4}],"overlapped":false,"timestamp":"2024-02-15T02:25:44.116Z"},{"category":"O2","operation":"Flip","direction":"horizontal","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":2,"x":2,"color":9},{"y":2,"x":1,"color":3},{"y":2,"x":0,"color":4},{"y":1,"x":2,"color":9},{"y":1,"x":1,"color":4},{"y":1,"x":0,"color":4},{"y":0,"x":2,"color":1},{"y":0,"x":1,"color":3},{"y":0,"x":0,"color":4}],"overlapped":false,"timestamp":"2024-02-15T02:25:44.676Z"},{"category":"Selection","operation":"SelectCell","position":{"x":2,"y":0},"grid":[[4,3,1],[4,4,9],[4,3,9]],"object":[],"overlapped":true,"timestamp":"2024-02-15T02:25:45.770Z"},{"category":"Selection","operation":"SelectCell","position":{"x":2,"y":0},"grid":[[4,3,0],[4,4,9],[4,3,9]],"object":[{"x":2,"y":0,"color":1}],"overlapped":true,"timestamp":"2024-02-15T02:25:48.009Z"},{"category":"Coloring","operation":"Paint","color":9,"grid":[[4,3,0],[4,4,9],[4,3,9]],"object":[{"x":2,"y":0,"color":9}],"overlapped":false,"timestamp":"2024-02-15T02:25:48.798Z"},{"category":"Selection","operation":"SelectCell","position":{"x":2,"y":0},"grid":[[4,3,9],[4,4,9],[4,3,9]],"object":[],"overlapped":true,"timestamp":"2024-02-15T02:25:49.697Z"},{"category":"Selection","operation":"SelectGrid","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":0,"x":0,"color":4},{"y":0,"x":1,"color":3},{"y":0,"x":2,"color":9},{"y":1,"x":0,"color":4},{"y":1,"x":1,"color":4},{"y":1,"x":2,"color":9},{"y":2,"x":0,"color":4},{"y":2,"x":1,"color":3},{"y":2,"x":2,"color":9}],"overlapped":true,"timestamp":"2024-02-15T02:25:52.142Z"},{"category":"Selection","operation":"SelectGrid","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":0,"x":0,"color":4},{"y":0,"x":1,"color":3},{"y":0,"x":2,"color":9},{"y":1,"x":0,"color":4},{"y":1,"x":1,"color":4},{"y":1,"x":2,"color":9},{"y":2,"x":0,"color":4},{"y":2,"x":1,"color":3},{"y":2,"x":2,"color":9}],"overlapped":true,"timestamp":"2024-02-15T02:25:52.158Z"},{"category":"O2","operation":"Flip","direction":"horizontal","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":0,"x":2,"color":4},{"y":0,"x":1,"color":3},{"y":0,"x":0,"color":9},{"y":1,"x":2,"color":4},{"y":1,"x":1,"color":4},{"y":1,"x":0,"color":9},{"y":2,"x":2,"color":4},{"y":2,"x":1,"color":3},{"y":2,"x":0,"color":9}],"overlapped":false,"timestamp":"2024-02-15T02:25:52.812Z"},{"category":"O2","operation":"Rotate","direction":"clockwise","grid":[[0,0,0],[0,0,0],[0,0,0]],"object":[{"y":2,"x":2,"color":4},{"y":1,"x":2,"color":3},{"y":0,"x":2,"color":9},{"y":2,"x":1,"color":4},{"y":1,"x":1,"color":4},{"y":0,"x":1,"color":9},{"y":2,"x":0,"color":4},{"y":1,"x":0,"color":3},{"y":0,"x":0,"color":9}],"overlapped":false,"timestamp":"2024-02-15T02:26:01.237Z"},{"category":"Selection","operation":"SelectCell","position":{"x":0,"y":2},"grid":[[9,9,9],[3,4,3],[4,4,4]],"object":[],"overlapped":true,"timestamp":"2024-02-15T02:26:02.288Z"},{"category":"Critical","operation":"Submit","grid":[[9,9,9],[3,4,3],[4,4,4]],"object":[],"overlapped":true,"special":{"succeed":false},"timestamp":"2024-02-15T02:26:10.990Z"}]''')
    
    # Generate SFT data
    sft_data = generate_sft_data(trajectory_data)
    
    # Display initial and final grids
    print("=== Initial and Final Grids ===")
    if sft_data:
        # Find the example with the full sequence
        full_sequence = max(sft_data, key=lambda x: x['sequence_length'])
        print(f"Initial grid:\n{format_grid(full_sequence['grid_before'])}\n")
        print(f"Final grid:\n{format_grid(full_sequence['grid_after'])}\n")
        print(f"Full action sequence: {full_sequence['output']}\n")
    
    # Save SFT data
    output_file = 'sft_data_74dd1130.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in sft_data:
            # Save only input and output for SFT
            sft_example = {
                'input': example['input'],
                'output': example['output']
            }
            f.write(json.dumps(sft_example, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(sft_data)} training examples")
    print(f"Saved to {output_file}")
    
    # Display some examples
    print("\n=== Sample Training Examples ===")
    for i, example in enumerate(sft_data[:5]):  # Show first 5 examples
        print(f"\nExample {i+1} (sequence length: {example['sequence_length']}):")
        print(f"Input:\n{example['input']}")
        print(f"Output: {example['output']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
