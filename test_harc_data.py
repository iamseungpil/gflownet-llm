import pandas as pd
import json

# h-arc data.csv 파일 읽기
harc_data_path = "data/h-arc/data/data.csv"
df = pd.read_csv(harc_data_path, nrows=1000)  # 처음 1000행만 읽기

# 데이터 구조 확인
print("H-ARC Data Structure:")
print(f"Total rows (sample): {len(df)}")
print(f"\nColumns: {list(df.columns)}")

# 6150a2bd 태스크 데이터 필터링
task_data = df[df['task_name'] == '6150a2bd']
print(f"\n\nData for task 6150a2bd:")
print(f"Number of actions: {len(task_data)}")

if len(task_data) > 0:
    print(f"Unique participants: {task_data['hashed_id'].nunique()}")
    print(f"\nAction types:")
    print(task_data['action'].value_counts())
    
    # 첫 번째 참가자의 action sequence 확인
    first_participant = task_data['hashed_id'].iloc[0]
    participant_data = task_data[task_data['hashed_id'] == first_participant]
    
    print(f"\n\nFirst participant's action sequence:")
    for idx, row in participant_data.iterrows():
        print(f"Action {row['action_id']}: {row['action']} - solved: {row['solved']}")
        if row['action_id'] > 10:  # 처음 10개만
            break
else:
    print("No data found for task 6150a2bd in the sample")

# re-arc 데이터도 확인
print("\n\n" + "="*60)
print("RE-ARC Data Structure:")
rearc_path = "data/re-arc/re_arc_extracted/re_arc/tasks/6150a2bd.json"
try:
    with open(rearc_path, 'r') as f:
        rearc_data = json.load(f)
    
    if isinstance(rearc_data, list):
        print(f"RE-ARC data is a list with {len(rearc_data)} examples")
        if rearc_data:
            print("\nFirst example:")
            print(f"Input shape: {len(rearc_data[0]['input'])}x{len(rearc_data[0]['input'][0])}")
            print(f"Output shape: {len(rearc_data[0]['output'])}x{len(rearc_data[0]['output'][0])}")
            print(f"Input: {rearc_data[0]['input']}")
            print(f"Output: {rearc_data[0]['output']}")
    else:
        print(f"RE-ARC data type: {type(rearc_data)}")
except Exception as e:
    print(f"Error reading RE-ARC data: {e}")
