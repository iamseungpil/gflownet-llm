# GFlowNet-LLM: ARC Task Augmentation Experiments

This repository contains experiments comparing different data augmentation strategies for solving ARC (Abstraction and Reasoning Corpus) tasks using Large Language Models.

## Overview

We compare three approaches:
1. **Baseline**: Using only original ARC training examples
2. **RE-ARC**: Using procedurally generated augmented examples
3. **H-ARC**: Using human-created augmented examples

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

This will run experiments on Task 178 (Diagonal Flip) and save results to `experiment_results.json`.

## Data Requirements

The script expects the following directory structure:
- `../../arc-prize-2024/arc-agi_training_challenges.json` - Original ARC data
- `../../re-arc/` - RE-ARC augmented data
- `../../h-arc/` - H-ARC augmented data

## Results

The experiment evaluates whether augmented training examples improve LLM performance on ARC tasks.

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

### Evaluation

The experiment measures:
- **Accuracy**: Whether the LLM produces the correct output grid
- **Data Efficiency**: How many additional examples are needed
- **Generalization**: Whether augmented data helps or hinders performance

## Citation

Based on the paper "Solution Augmentation for ARC-AGI Problems Using GFlowNet: A Probabilistic Exploration Approach"
