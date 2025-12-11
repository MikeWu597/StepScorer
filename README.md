# 📊 StepScorer

[English](README.md) | [中文](README_zh.md)

StepScorer is a machine learning project designed for text scoring. The model can be applied to various specific tasks—such as evaluating grammatical correctness, detecting discriminatory content, and other text assessment tasks—depending on the training data provided. Its usage is highly flexible and supports a wide range of scoring applications. This repository provides tools for training the model, performing inference, and visualizing results.

Unlike traditional approaches that produce a single holistic evaluation, StepScorer adopts a decomposed strategy, generating interpretable step-by-step assessments while preserving the contextual relationship across the entire sequence. During each inference step, the model outputs delta values representing the contribution of the current step to the total score. As the sequence progresses, these delta values gradually decrease and approach zero, indicating that the model is becoming increasingly confident about the final score. The core innovation of this approach lies in its incremental scoring mechanism: the delta values diminish over time steps and converge toward zero, with their cumulative sum approaching a limit—the model’s predicted final score. This mathematical principle underpins both the training and inference processes.

The model is lightweight. During development, a dataset containing approximately 1,500 discriminatory statements was used for training and inference on an Nvidia GeForce RTX 5060 Ti GPU. Training took about 40 seconds, and a single inference operation required roughly 1 second.

## 🔍 Example

Target for detection: **The sentence contains race-color related discriminatory content.**

| Input                                               | Output |
|----------------------------------------------------|--------|
| PolyU is an outstanding university.                | 0.401  |
| White Americans aren't smart enough to be scientists. | 4.912  |

When the output value exceeds a certain threshold, the input is considered to contain racially discriminatory content.

## 📁 Table of Contents
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Debugging](#debugging)
- [Model Architecture](#model-architecture)

## 🗃️ Dataset Preparation

Before training or running inference, you need to format your dataset appropriately:

The project expects a CSV file with three columns: `standard`, `object`, and `score`.

Example data structure:

```markdown
data/
└── sample_data.csv
```

The repository includes two ready-to-use training datasets for detecting discriminatory content, adapted from [CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models](https://aclanthology.org/2020.emnlp-main.154/) (Nangia et al., EMNLP 2020). Users may choose either as needed.

Each row in the CSV should contain:
- `standard`: the evaluation criterion
- `object`: the item being evaluated
- `score`: the ground-truth score (on a 0–5 scale, or another consistent scale)

Example row:
```markdown
standard,object,score
"Poetry should rhyme and convey emotional depth.","Autumn leaves drift; water flows alone. On a solitary boat, I recall old friends. Ten years—life and death—vast and茫. Not thought of, yet unforgettable.",4.9
```

## 🔧 Dependency Setup

All required packages are listed in [requirements.txt](requirements.txt). Install them using:

```bash
pip install -r requirements.txt
```

## 🏋️ Training

To train the StepScorer model, directly modify the `CONFIG` dictionary in [train.py](train.py):

Key training configuration options in `CONFIG`:
- `data_path`: path to the training data (default: `'data/sample_data.csv'`)
- `model_save_path`: file path to save the trained model (default: `'scoring_model.pt'`)
- `epochs`: number of training epochs (default: `20`)
- `batch_size`: training batch size (default: `24`)
- `lr`: learning rate for the optimizer (default: `0.002`)

During training, the model will:
1. Load and preprocess the training data
2. Initialize model parameters
3. Run the specified number of training loops
4. Save the best model based on validation loss

Simply run:
```bash
python train.py
```

## 🔮 Inference

After training, you can use the [inference.py](inference.py) script to score new sequences.

Before running inference, update the `CONFIG` dictionary in [inference.py](inference.py):
- `model_path`: path to the trained model checkpoint (default: `'scoring_model.pt'`)

Then run:
```bash
python inference.py
```

The inference script will:
1. Load the trained model
2. Prompt for input of a standard and an object
3. Generate stepwise scores for the sequence
4. Save the results in a structured format to `'scoring_steps.json'`

## 🐞 Debugging

For debugging and visualization purposes, use the [figure.py](figure.py) script to convert model outputs into plots:

First, ensure you have a scoring result file (generated by `inference.py`):
- By default, this will be `'scoring_steps.json'`

Then run:
```bash
python figure.py
```

This script helps:
- Visualize prediction results
- Display the cumulative score evolution over steps
- Show delta values at each step
- Generate charts for analysis and presentation

Visualization results will be saved as `'scoring_evolution.png'`.

## 🧩 Model Architecture

The model architecture is defined in [model.py](model.py). Key components include:

### Core Architecture
- **BERT Encoder**: Uses a pretrained BERT model to process input text (frozen)
- **GRU Module**: Models the sequential scoring process
- **Delta Predictor**: Predicts incremental score changes at each step
- **Accumulator**: Computes cumulative scores from deltas

### Key Features
- Uses BERT for semantic understanding of both standard and object
- Employs a GRU to model the step-by-step scoring process
- Predicts score deltas at each step rather than absolute scores

### Model Parameters
Main hyperparameters adjustable via the respective `CONFIG` dictionaries:
- Hidden dimension size (default: `128`)
- Maximum number of steps (default: `100`)
- Learning rate (default: `0.002`)
- Batch size (default: `24`)

For implementation details, please refer to [model.py](model.py).

## 🤝 Legal Notice

- The source code, documentation, and related technical materials (collectively referred to as “the Technology”) made publicly available in this project are provided solely for technical demonstration and academic exchange purposes. They do not constitute a waiver, transfer, or license of any intellectual property rights related to the Technology.
- The developer/applicant reserves all rights to file for patents, trademarks, copyrights, or other intellectual property protections for the Technology in the People’s Republic of China and other jurisdictions.
- No individual or organization may claim that the Technology has entered the public domain based on this disclosure, nor may they use the Technology for commercial purposes, patent design-around, or any activity that impedes the patentability of the Technology without prior written authorization.
- This disclosure does not grant any express or implied license to any third party. All rights are reserved.