# GFlowNet-LLM: ARC Task Augmentation Experiments

This repository contains experiments comparing different data augmentation strategies for solving ARC (Abstraction and Reasoning Corpus) tasks using Large Language Models.

## Overview

We compare three approaches:
1. **Baseline**: Using only original ARC training examples
2. **RE-ARC**: Using procedurally generated augmented examples
3. **H-ARC**: Using human-created augmented examples

## New Features

### 1. Data Scaling Analysis
- Track accuracy changes as the number of augmented examples increases
- Measure accuracy delta (Δ) between each data point

### 2. Multi-Candidate Evaluation
- Generate multiple candidates (default: 5) per configuration
- Calculate average accuracy for more reliable results

### 3. Data Validation
- Verify that RE-ARC and H-ARC data correspond to the correct task
- Check data format and transformation consistency

### 4. W&B Integration
- Real-time visualization of accuracy trends
- Track performance across different data sizes

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Basic Experiment (Original)
```bash
python main.py
```

### Advanced Data Scaling Experiment
```bash
python main_v2.py
```

### Validate Augmentation Data
```bash
python validate_data.py
```

## Data Requirements

The script expects the following directory structure:
- `../../arc-prize-2024/arc-agi_training_challenges.json` - Original ARC data
- `../../re-arc/` - RE-ARC augmented data
- `../../h-arc/` - H-ARC augmented data

## Experiment Details

### Task 178 (6150a2bd): Diagonal Flip
This is a whole-grid transformation task where the solution requires:
1. Rotating the grid
2. Flipping horizontally or vertically
3. Submitting the final answer

### Comparison Methods

1. **Baseline**: Only uses the original 2-3 training examples from ARC
2. **RE-ARC**: Adds procedurally generated examples that follow the same transformation rule
3. **H-ARC**: Adds human-created examples with variations

### Evaluation Metrics

1. **Accuracy**: Whether the LLM produces the correct output grid
2. **Data Efficiency**: How accuracy changes with additional examples
3. **Reliability**: Average accuracy across multiple candidates
4. **Generalization**: Whether augmented data helps or hinders performance

## Results Visualization

The experiment logs results to Weights & Biases (W&B) for real-time visualization:
- Accuracy trends as data size increases
- Comparison between different augmentation methods
- Accuracy deltas (rate of improvement)

## Output Files

- `experiment_results.json`: Basic experiment results
- `scaling_experiment_results.json`: Detailed results with accuracy by data size
- W&B dashboard: Interactive visualizations

## Citation

Based on the paper "Solution Augmentation for ARC-AGI Problems Using GFlowNet: A Probabilistic Exploration Approach"
