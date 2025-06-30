# O2ARC Action Prediction SFT Experiment

이 프로젝트는 O2ARC trajectory 데이터를 사용하여 LLM이 grid 변환 action을 예측하도록 SFT(Supervised Fine-Tuning)하는 실험입니다.

## 프로젝트 구조

- `o2arc_data_preparation.py`: Trajectory 데이터를 SFT용 학습 데이터로 변환
- `o2arc_experiment.py`: Llama-3.1 모델을 SFT하는 메인 실험 코드
- `requirements.txt`: 필요한 패키지 목록
- `run_experiment.sh`: 전체 실험 실행 스크립트

## 실험 개요

### 데이터 생성 전략
단계별로 데이터를 생성하여 모델이 점진적으로 학습할 수 있도록 합니다:
1. **1-action 예측**: 하나의 action 전후 grid를 보고 action 맞추기
2. **2-action 예측**: 두 개의 action 전후 grid를 보고 action 시퀀스 맞추기
3. **N-action 예측**: 전체 trajectory까지 확장

### 예시
```
Input:
Initial grid:
9 3 4
9 4 4
9 3 4

Final grid:
9 9 9
3 4 3
4 4 4

What action(s) transformed the initial grid to the final grid?

Output: Flip(horizontal) -> Rotate(clockwise)
```

## 설치 및 실행

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
python o2arc_data_preparation.py
```

### 3. SFT 실행
```bash
# 기본 설정으로 실행
bash run_experiment.sh

# 또는 커스텀 설정으로 실행
python o2arc_experiment.py \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --num_train_epochs 5 \
    --learning_rate 1e-4
```

## 주요 파라미터

### 모델 설정
- `--use_4bit`: 4-bit 양자화 사용 (메모리 절약)
- `--use_lora`: LoRA 사용 (효율적인 파인튜닝)
- `--lora_r`: LoRA rank (기본값: 16)

### 학습 설정
- `--num_train_epochs`: 학습 epoch 수
- `--learning_rate`: 학습률
- `--per_device_train_batch_size`: 배치 크기

### 프롬프트 템플릿
- `llama3`: Llama-3.1 instruction 형식
- `custom`: 커스텀 형식

## 실험 결과 확인

1. **WandB**: 학습 과정 모니터링
2. **체크포인트**: `./o2arc-llama3-sft` 폴더에 저장
3. **평가 예시**: 학습 후 자동으로 출력

## 추가 실험 아이디어

1. **다양한 문제**: 여러 ARC 문제에 대해 실험
2. **Action 복잡도**: Action 시퀀스 길이별 성능 분석
3. **프롬프트 엔지니어링**: 다양한 프롬프트 형식 비교
4. **모델 크기**: 다른 크기의 Llama 모델 비교

## 참고사항

- GPU 메모리가 부족한 경우 `--per_device_train_batch_size`를 줄이거나 `--gradient_accumulation_steps`를 늘리세요
- Llama-3.1 모델 접근 권한이 필요합니다 (HuggingFace 로그인 필요)
