# GFlowNet-LLM: ARC Task Augmentation Experiments

이 프로젝트는 다양한 데이터 증강 전략을 사용하여 LLM의 ARC (Abstraction and Reasoning Corpus) 문제 해결 성능을 비교 분석합니다.

## 📌 프로젝트 개요

**목표**: 다음 세 가지 접근법의 효과를 비교
1. **Baseline**: 원본 ARC 훈련 예제만 사용
2. **RE-ARC**: 절차적으로 생성된 증강 예제 사용
3. **H-ARC**: 인간의 문제 해결 action trace 활용

## 🗂️ 정리된 파일 구조

### 핵심 실험 파일
- **`experiment_rearc.py`**: RE-ARC 증강 실험 (절차적 생성 데이터)
- **`experiment_harc.py`**: H-ARC 실험 (인간 행동 데이터 + action sequence 예측)
- **`run_experiments.py`**: 모든 실험을 통합 실행하는 메인 오케스트레이터
- **`multi_model_experiment.py`**: 다중 모델 비교 (Llama, OpenAI O1, Gemma, Qwen)

### 유틸리티 파일  
- **`arc.py`**: ARC 데이터 로딩 및 처리
- **`download_datasets.py`**: 데이터셋 다운로드 및 설정
- **`validate_data.py`**: 데이터 무결성 검증
- **`plot_results.py`**: 결과 시각화

## 🚀 사용 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
pip install -r requirements_multi_model.txt  # 멀티모델 실험용
```

### 2. 데이터 준비
```bash
python download_datasets.py    # 데이터셋 다운로드
python validate_data.py        # 데이터 검증
```

### 3. 실험 실행

#### 기본 실험 (RE-ARC + H-ARC)
```bash
python run_experiments.py
```

#### 개별 실험
```bash
# RE-ARC 실험만
python experiment_rearc.py

# H-ARC 실험만 (그리드 생성 + action sequence 예측)
python experiment_harc.py
```

#### 멀티모델 비교 실험
```bash
# 빠른 테스트
python multi_model_experiment.py quick

# 전체 실험
python multi_model_experiment.py
```

## 🔬 실험 유형

### 1. RE-ARC 실험
- **목적**: 더 많은 예제를 통한 패턴 학습 효과 측정
- **데이터**: 각 태스크당 최대 1000개의 생성된 입출력 쌍
- **변수**: 증강 예제 개수 (0, 1, 2, 3, 5, 10, 20)

### 2. H-ARC 실험
#### a) 그리드 생성
- **목적**: 인간의 해결 과정이 LLM 성능 향상에 미치는 영향
- **프롬프트 타입**:
  - `full_trace`: 상세한 단계별 행동 시퀀스
  - `hint`: 요약된 행동 패턴 힌트
- **변수**: 인간 trace 개수 (0, 1, 3, 5)

#### b) Action Sequence 예측
- **목적**: LLM이 인간과 유사한 문제 해결 과정을 학습하는지 측정
- **평가**: 예측된 action sequence와 실제 인간 sequence의 유사도
- **메트릭**: Exact match, Sequence similarity, Common actions ratio

### 3. 멀티모델 비교
- **모델**: Llama 3.1-8B, OpenAI O1, Google Gemma 7B, Qwen 2.5-7B
- **목적**: 모델별 기본 성능과 증강 데이터 활용 능력 비교

## 📊 결과 파일

실험 완료 후 다음 파일들이 생성됩니다:
- `rearc_results.json`: RE-ARC 실험 결과
- `harc_grid_results.json`: H-ARC 그리드 생성 결과  
- `harc_action_results.json`: H-ARC action sequence 예측 결과
- `harc_combined_results.json`: H-ARC 통합 결과
- `multi_model_results.json`: 멀티모델 비교 결과
- `experiment_summary.md`: 결과 요약 보고서
- `comparison_*.png`: 태스크별 비교 차트
- `overall_comparison.png`: 전체 성능 비교

## 🎯 주요 연구 질문

1. **더 많은 예제 vs 해결 과정**: RE-ARC의 추가 예제와 H-ARC의 인간 행동 데이터 중 어느 것이 더 효과적인가?
2. **모델별 차이**: 어떤 LLM이 ARC 문제 해결에 가장 적합한가?
3. **데이터 효율성**: 각 접근법에서 최적의 데이터 양은 얼마인가?
4. **Action 학습**: LLM이 인간의 문제 해결 과정을 얼마나 잘 모방할 수 있는가?

## 📈 W&B 모니터링

실험 과정은 Weights & Biases로 실시간 추적됩니다:
- 정확도 변화 추이
- 모델별/방법별 성능 비교  
- 하이퍼파라미터 효과 분석

## 🔧 설정 옵션

### 실험 파라미터
- `num_candidates`: 각 설정당 생성할 후보 답안 수 (기본: 5)
- `use_colors`: 숫자 대신 색상 이름 사용 여부 (기본: False)
- `task_ids`: 실험할 태스크 목록 (기본: ["6150a2bd", "178fcbfb", "1190e5a7", "150deff5"])

### API 키 설정
```bash
# 환경변수로 설정 (권장)
export OPENAI_API_KEY="your-openai-api-key"
export WANDB_API_KEY="your-wandb-api-key"
```

## 📋 데이터셋 정보

### H-ARC (Human ARC)
- **위치**: `data/h-arc/`
- **내용**: 인간 참가자들의 ARC 문제 해결 과정
- **포함**: Action traces, 시간 정보, 성공/실패 데이터

### RE-ARC (Reverse-Engineering ARC)  
- **위치**: `data/re-arc/`
- **내용**: 절차적으로 생성된 ARC 변형 문제
- **규모**: 각 태스크당 1000개의 추가 예제

## 🤝 기여 및 인용

이 프로젝트는 다음 논문을 기반으로 합니다:
"Solution Augmentation for ARC-AGI Problems Using GFlowNet: A Probabilistic Exploration Approach"

---

**Note**: 이 README는 코드 정리 후 업데이트된 버전입니다. 이제 총 8개의 핵심 파일로 구성되어 있으며, 중복 제거와 명확한 실행 경로를 제공합니다.