#!/usr/bin/env python3
"""
Script to download and setup re-arc and h-arc datasets
"""

import os
import subprocess
import sys
import zipfile
import json
from pathlib import Path

def run_command(cmd, cwd=None):
    """Execute a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            print(f"Error executing: {cmd}")
            print(f"Error: {result.stderr}")
            return False
        print(f"Success: {cmd}")
        return True
    except Exception as e:
        print(f"Exception running command: {e}")
        return False

def download_datasets():
    """Download re-arc and h-arc datasets"""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("=== Downloading re-arc dataset ===")
    # Clone re-arc repository
    if not (data_dir / "re-arc").exists():
        if run_command("git clone https://github.com/michaelhodel/re-arc.git", cwd="data"):
            print("✓ re-arc repository cloned successfully")
        else:
            print("✗ Failed to clone re-arc repository")
            return False
    else:
        print("✓ re-arc directory already exists")
    
    print("\n=== Downloading h-arc dataset ===")
    # Clone h-arc repository  
    if not (data_dir / "h-arc").exists():
        if run_command("git clone https://github.com/Le-Gris/h-arc.git", cwd="data"):
            print("✓ h-arc repository cloned successfully")
        else:
            print("✗ Failed to clone h-arc repository")
            return False
    else:
        print("✓ h-arc directory already exists")
    
    # Extract re-arc.zip if it exists
    re_arc_zip = data_dir / "re-arc" / "re_arc.zip"
    if re_arc_zip.exists():
        print("\n=== Extracting re-arc.zip ===")
        try:
            with zipfile.ZipFile(re_arc_zip, 'r') as zip_ref:
                extract_path = data_dir / "re-arc" / "re_arc_extracted"
                zip_ref.extractall(extract_path)
            print(f"✓ Extracted re_arc.zip to {extract_path}")
        except Exception as e:
            print(f"✗ Failed to extract re_arc.zip: {e}")
    
    print("\n=== Dataset Structure ===")
    print("\n1. re-arc dataset:")
    print("   - Location: data/re-arc/")
    print("   - Contains procedural generators for 400 ARC training tasks")
    print("   - re_arc.zip has 1000 generated examples per task")
    print("   - Each example has 'input' and 'output' grids")
    
    print("\n2. h-arc dataset:")
    print("   - Location: data/h-arc/")
    print("   - Contains human action traces for solving ARC tasks")
    print("   - Includes timing, actions, and error data")
    print("   - Need to download data from OSF repository separately")
    
    return True

def create_arc_data_loader():
    """Create a Python module to load ARC data in the format expected by your code"""
    
    loader_code = '''"""
ARC Data Loader Module
Provides functions to load ARC problems from re-arc and h-arc datasets
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

class ARCProblem:
    """ARC Problem representation matching the expected format"""
    def __init__(self, uid: str, train_pairs: List, test_pairs: List):
        self.uid = uid
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs

class GridPair:
    """Input-output grid pair"""
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.array(x)
        self.y = np.array(y)

def load_original_arc_tasks(task_type: str = "training") -> Dict[str, Dict]:
    """Load original ARC tasks from re-arc dataset"""
    base_path = Path("data/re-arc/arc_original")
    
    if not base_path.exists():
        # Try extracting from zip
        import zipfile
        zip_path = Path("data/re-arc/arc_original.zip")
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(Path("data/re-arc"))
    
    tasks = {}
    task_dir = base_path / task_type
    
    if task_dir.exists():
        for json_file in task_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                task_data = json.load(f)
                tasks[json_file.stem] = task_data
    
    return tasks

def load_rearc_generated_examples(task_id: str, num_examples: Optional[int] = None) -> List[Dict]:
    """Load generated examples from re-arc dataset"""
    
    # First check extracted directory
    extracted_path = Path(f"data/re-arc/re_arc_extracted/re_arc/tasks/{task_id}.json")
    
    if not extracted_path.exists():
        # Try to extract from zip
        import zipfile
        zip_path = Path("data/re-arc/re_arc.zip")
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                try:
                    # Extract specific file
                    zip_ref.extract(f"re_arc/tasks/{task_id}.json", Path("data/re-arc/re_arc_extracted"))
                except KeyError:
                    print(f"Task {task_id} not found in re_arc.zip")
                    return []
    
    if extracted_path.exists():
        with open(extracted_path, 'r') as f:
            examples = json.load(f)
            if num_examples:
                return examples[:num_examples]
            return examples
    
    return []

def convert_to_arc_problems(task_type: str = "training") -> List[ARCProblem]:
    """Convert ARC tasks to ARCProblem format expected by the training code"""
    
    tasks = load_original_arc_tasks(task_type)
    problems = []
    
    for task_id, task_data in tasks.items():
        train_pairs = []
        for example in task_data.get('train', []):
            pair = GridPair(
                x=np.array(example['input']),
                y=np.array(example['output'])
            )
            train_pairs.append(pair)
        
        test_pairs = []
        for example in task_data.get('test', []):
            pair = GridPair(
                x=np.array(example['input']),
                y=np.array(example['output'])  
            )
            test_pairs.append(pair)
        
        problem = ARCProblem(
            uid=task_id,
            train_pairs=train_pairs,
            test_pairs=test_pairs
        )
        problems.append(problem)
    
    return problems

# Create module-level variables matching the expected format
train_problems = convert_to_arc_problems("training")
validation_problems = convert_to_arc_problems("evaluation")

# Add function to get re-arc generated examples as additional training data
def get_rearc_augmented_problems(original_problems: List[ARCProblem], 
                                examples_per_task: int = 10) -> List[ARCProblem]:
    """Augment original problems with re-arc generated examples"""
    
    augmented_problems = []
    
    for problem in original_problems:
        # Get generated examples
        generated = load_rearc_generated_examples(problem.uid, examples_per_task)
        
        if generated:
            # Create new train pairs including generated examples
            new_train_pairs = list(problem.train_pairs)  # Copy original
            
            for gen_example in generated:
                pair = GridPair(
                    x=np.array(gen_example['input']),
                    y=np.array(gen_example['output'])
                )
                new_train_pairs.append(pair)
            
            # Create augmented problem
            augmented = ARCProblem(
                uid=f"{problem.uid}_augmented",
                train_pairs=new_train_pairs,
                test_pairs=problem.test_pairs
            )
            augmented_problems.append(augmented)
        else:
            # If no generated examples, use original
            augmented_problems.append(problem)
    
    return augmented_problems

print(f"Loaded {len(train_problems)} training problems")
print(f"Loaded {len(validation_problems)} validation problems")
'''
    
    # Write the loader module
    loader_path = Path("arc.py")
    with open(loader_path, 'w') as f:
        f.write(loader_code)
    
    print(f"\n✓ Created arc.py data loader module")
    print("  You can now import it with: from arc import train_problems, validation_problems")
    
    return True

if __name__ == "__main__":
    print("Starting dataset download...")
    
    if download_datasets():
        print("\n✅ Datasets downloaded successfully!")
        
        # Create the data loader
        if create_arc_data_loader():
            print("\n✅ Data loader created successfully!")
            
            # Test the loader
            print("\n=== Testing data loader ===")
            try:
                from arc import train_problems, validation_problems
                print(f"✓ Successfully loaded {len(train_problems)} training problems")
                print(f"✓ Successfully loaded {len(validation_problems)} validation problems")
                
                if train_problems:
                    sample = train_problems[0]
                    print(f"\nSample problem:")
                    print(f"  - UID: {sample.uid}")
                    print(f"  - Train pairs: {len(sample.train_pairs)}")
                    print(f"  - Test pairs: {len(sample.test_pairs)}")
                    print(f"  - Input shape: {sample.train_pairs[0].x.shape}")
                    print(f"  - Output shape: {sample.train_pairs[0].y.shape}")
                    
            except Exception as e:
                print(f"✗ Error testing loader: {e}")
    else:
        print("\n❌ Failed to download datasets")
        sys.exit(1)
