# Adaptive Precision Training for Multi-Modal Models

## Overview

This project implements and evaluates an adaptive precision training strategy for a multi-modal neural network using the DeepSpeed library. The goal is to optimize training performance (throughput, memory usage) while maintaining model accuracy by dynamically switching between FP32 and FP16 precision based on training stability metrics. The framework supports running experiments across different configurations (e.g., batch sizes, GPU types) and analyzing the results.

The core model is a `FusionModel` that processes three modalities: text, image, and graph data.

## Features

* **Multi-Modal Model:** A `FusionModel` combining text (`TextEncoder`), image (`ImageEncoder`), and graph (`GraphEncoder`) inputs 
* **DeepSpeed Integration:** Utilizes DeepSpeed for efficient distributed training (configured for single GPU in `run_experiments.py` but uses DeepSpeed API).
* **Adaptive Precision Training:** Implements a custom callback (`AdaptivePrecisionCallback`) to monitor loss variance and trend, dynamically switching between FP16 and FP32 during training for optimal performance and stability.
* **Experiment Management:** Script (`run_experiments.py`) to systematically run training jobs with varying batch sizes, precision modes (FP32, FP16, Adaptive), and epochs. Supports targeting specific GPU types (e.g., A100, V100).
* **Results Analysis:** Script (`analyze_results.py`) to parse experiment logs, calculate performance metrics (throughput, memory, loss), generate comparison plots, and create an HTML summary report.

## File Structure

* `main.py`: Main entry point to run experiments or analysis via command-line arguments.
* `run_experiments.py`: Handles the orchestration of running multiple training configurations to test and see the results.
* `train.py`: Contains the primary training loop, model initialization, DeepSpeed setup, adaptive precision logic, evaluation, and metric logging.
* `analyze_results.py`: Loads, processes, and visualizes results from completed experiments.
* `model.py`: Defines the neural network architectures for text, image, graph encoders and the final fusion model.
* `dataloader.py`: Defines the data loading and preprocessing pipeline for the *training* datasets.
* `test_dataloader.py`: Defines the data loading and preprocessing pipeline for the *test* datasets.
* `datasetcleaning.py`: Contains dataset transformation classes and the `MultiModalDataset` class.
* `ds_config.json`: Base DeepSpeed configuration file (partially used/overridden in `train.py`).

## Setup

1.  **Clone the repository.**
2.  **Install Dependencies:** Ensure Python, PyTorch, DeepSpeed, and other required libraries (`torchtext`, `torchvision`, `ogb`, `pandas`, `matplotlib`, `numpy`) are installed.
3.  **CUDA Environment:** The `train.py` script sets the `CUDA_HOME` environment variable and creates a dummy `nvcc` file if needed; ensure your CUDA installation is compatible.

## Data

The project uses subsets of the following datasets:
* **Text:** IMDB dataset (via `torchtext`) 
* **Image:** CIFAR10 dataset (via `torchvision`) 
* **Graph:** OGBG-MOLHIV dataset (via `ogb`) 

Data will be downloaded automatically to a `data/` subdirectory within the project folder upon the first run

### 1. Run Experiments

Execute training runs with different configurations. Experiments will be saved in subdirectories under the specified output directory.

```bash
python main.py run_experiments --output_dir ./my_experiments --batch_sizes 32 64 --epochs 10 --gpu_types A100
