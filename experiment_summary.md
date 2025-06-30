# ARC-LLM Augmentation Experiments Summary

## 1. RE-ARC Experiment Results

RE-ARC uses procedurally generated input-output grid pairs for augmentation.

### Task 6150a2bd
- Number of available RE-ARC examples: 1000
- Results by augmentation size:

| Augmented Examples | Accuracy | Correct/Total |
|-------------------|----------|---------------|
| 0 | 0.00% | 0/5 |
| 1 | 0.00% | 0/5 |
| 2 | 0.00% | 0/5 |
| 3 | 0.00% | 0/5 |
| 5 | 0.00% | 0/5 |
| 10 | 0.00% | 0/5 |

### Task 178fcbfb
- Number of available RE-ARC examples: 1000
- Results by augmentation size:

| Augmented Examples | Accuracy | Correct/Total |
|-------------------|----------|---------------|
| 0 | 0.00% | 0/5 |
| 1 | 0.00% | 0/5 |
| 2 | 0.00% | 0/5 |
| 3 | 0.00% | 0/5 |
| 5 | 0.00% | 0/5 |
| 10 | 0.00% | 0/5 |

### Task 1190e5a7
- Number of available RE-ARC examples: 1000
- Results by augmentation size:

| Augmented Examples | Accuracy | Correct/Total |
|-------------------|----------|---------------|
| 0 | 0.00% | 0/5 |
| 1 | 0.00% | 0/5 |
| 2 | 0.00% | 0/5 |
| 3 | 0.00% | 0/5 |
| 5 | 0.00% | 0/5 |
| 10 | 0.00% | 0/5 |

### Task 150deff5
- Number of available RE-ARC examples: 1000
- Results by augmentation size:

| Augmented Examples | Accuracy | Correct/Total |
|-------------------|----------|---------------|
| 0 | 0.00% | 0/5 |
| 1 | 0.00% | 0/5 |
| 2 | 0.00% | 0/5 |
| 3 | 0.00% | 0/5 |
| 5 | 0.00% | 0/5 |
| 10 | 0.00% | 0/5 |


## 2. H-ARC Experiment Results

H-ARC uses human action traces for providing solution hints.

### Task 6150a2bd
- Number of available human traces: 0

#### Full Trace Prompt

| Human Traces | Accuracy | Correct/Total |
|--------------|----------|---------------|
| 0 | 0.00% | 0/5 |

#### Hint Prompt

| Human Traces | Accuracy | Correct/Total |
|--------------|----------|---------------|
| 0 | 0.00% | 0/5 |

### Task 178fcbfb
- Number of available human traces: 0

#### Full Trace Prompt

| Human Traces | Accuracy | Correct/Total |
|--------------|----------|---------------|
| 0 | 0.00% | 0/5 |

#### Hint Prompt

| Human Traces | Accuracy | Correct/Total |
|--------------|----------|---------------|
| 0 | 0.00% | 0/5 |

### Task 1190e5a7
- Number of available human traces: 0

#### Full Trace Prompt

| Human Traces | Accuracy | Correct/Total |
|--------------|----------|---------------|
| 0 | 0.00% | 0/5 |

#### Hint Prompt

| Human Traces | Accuracy | Correct/Total |
|--------------|----------|---------------|
| 0 | 0.00% | 0/5 |

### Task 150deff5
- Number of available human traces: 0

#### Full Trace Prompt

| Human Traces | Accuracy | Correct/Total |
|--------------|----------|---------------|
| 0 | 0.00% | 0/5 |

#### Hint Prompt

| Human Traces | Accuracy | Correct/Total |
|--------------|----------|---------------|
| 0 | 0.00% | 0/5 |


## 3. Key Findings

### Best Performance by Method

| Task ID | RE-ARC Best | H-ARC Best | Better Method |
|---------|-------------|------------|---------------|
| 6150a2bd | 0.00% | 0.00% | Tie |
| 178fcbfb | 0.00% | 0.00% | Tie |
| 1190e5a7 | 0.00% | 0.00% | Tie |
| 150deff5 | 0.00% | 0.00% | Tie |

### Observations

1. **RE-ARC Performance**: Grid-based augmentation shows [describe pattern]
2. **H-ARC Performance**: Action trace guidance shows [describe pattern]
3. **Data Efficiency**: [Compare how many examples needed for good performance]
4. **Prompt Strategy**: [Compare full trace vs hint effectiveness]
