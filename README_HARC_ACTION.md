# H-ARC Action Sequence 실험 가이드

## 개요

H-ARC 실험을 두 가지 방식으로 수행합니다:
1. **그리드 생성**: 인간의 action trace를 힌트로 사용하여 최종 그리드를 직접 생성
2. **Action Sequence 예측**: 인간이 문제를 해결한 action sequence를 예측

## 실행 방법

### 1. Action Sequence 실험만 실행
```bash
python run_harc_action_only.py
```

### 2. 특정 태스크로 실험
```bash
python run_harc_action_only.py --tasks 6150a2bd 178fcbfb
```

### 3. 그리드 생성 + Action Sequence 실험 모두 실행
```bash
python run_harc_action_only.py --combined
```

### 4. 모든 실험 실행 (RE-ARC + H-ARC 그리드 + H-ARC Action)
```bash
python run_experiments_with_action.py
```

## 실험 구조

### H-ARC Action Sequence 예측
- **입력**: 원본 training examples + 인간의 action sequence 예시
- **출력**: 테스트 케이스를 해결하기 위한 action sequence
- **평가 지표**:
  - Sequence similarity (SequenceMatcher)
  - Common actions ratio
  - LCS (Longest Common Subsequence) ratio
  - Exact match rate

### 예시 프롬프트
```
Training Examples:
Example 1:
Input: [[3,3,8],[3,7,0],[5,0,0]]
Output: [[0,0,5],[0,7,3],[8,3,3]]

How humans solved this task (example action sequences):

Human Solution 1:
1. click_cell
2. select_color (color: 3)
3. flood_fill
4. copy
5. paste
6. rotate
7. submit

Now predict the action sequence for this test case:
Test Input: [[6,3,5],[6,8,0],[4,0,0]]
Expected Output: [[0,0,4],[0,8,6],[5,3,6]]

List the actions needed to transform the input to the output.
```

## 결과 파일

- `harc_action_sequence_results.json`: Action sequence 예측 결과
- `harc_grid_results.json`: 그리드 생성 결과
- `harc_combined_results.json`: 두 실험의 통합 결과
- `experiment_summary_all.md`: 모든 실험의 요약 보고서

## 주요 메트릭

### Action Sequence 예측
- **Average Similarity**: 예측된 sequence와 실제 인간 sequence의 유사도
- **Common Actions Ratio**: 공통 action의 비율
- **Exact Match Rate**: 완전히 일치하는 sequence의 비율

### 비교 분석
- RE-ARC vs H-ARC Grid: 증강 데이터 vs 인간 힌트의 효과
- H-ARC Grid vs H-ARC Action: 직접 생성 vs 과정 예측의 차이

## 파일 구조
```
experiment_harc.py           # 기존 H-ARC 그리드 생성 실험
experiment_harc_action.py    # 새로운 H-ARC action sequence 예측 실험
run_harc_action_only.py      # Action sequence 실험 실행 스크립트
run_experiments_with_action.py # 모든 실험 통합 실행
```

## 필요 라이브러리
```bash
pip install pandas python-Levenshtein
```
