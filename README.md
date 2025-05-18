<div align="center">
  <img src="https://img.shields.io/badge/language-Python-%233776AB.svg?logo=python">
  <img src="https://img.shields.io/badge/uses-Deep%20Neural%20Network-%232A2F3D.svg">
  <img src="https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg" alt="Uses: Hugging Face">
</div>

<div align="center">
   <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
   <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white" />
   <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
</div>

# <p align="center"> Brain Tumor Classification and Detection <br> Using Machine Learning </p>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#features">Features</a></li>
    <li>
      <a href="#workflow">Workflow</a>
      <ul>
        <li><a href="#workflow-configuration">Workflow Configuration</a></li>
        <li><a href="#model-selection">Model Selection</a></li>
      </ul>
    </li>
    <li>
      <a href="#key-components">Key Components</a>
      <ul>
        <li><a href="#data-loading-and-handling">Data Loading and Handling</a></li>
        <li><a href="#data-analysis-and-exploration">Data Analysis and Exploration</a></li>
        <li><a href="#image-processing-and-visualization">Image Processing and Visualization</a></li>
        <li><a href="#model-application">Model Application</a></li>
        <li><a href="#performance-comparison">Performance Comparison</a></li>
      </ul>
    </li>
    <li>
      <a href="#model-comparison-classification-vs-detection">Model Comparison: Classification vs. Detection</a>
      <ul>
        <li><a href="#visual-comparison">Visual Comparison</a></li>
        <li><a href="#additional-detection-model-outputs">Additional Detection Model Outputs</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a>
      <ul>
        <li><a href="#non-tumor-examples">Non-Tumor Examples</a></li>
        <li><a href="#tumor-examples">Tumor Examples</a></li>
      </ul>
    </li>
  </ol>
</details>

## Overview

This project implements advanced machine learning techniques for the detection and classification of brain tumors using medical imaging data. It uses object classification and object detection models to for brain MRI scans analysis.

## Features

1. **Data Processing**: Utilizes the Mahadih534 brain tumor dataset for training and evaluation.
2. **Model Implementation**: Includes both classification and detection models for comprehensive tumor analysis.
3. **Visualization**: Generates output images for model analysis, tumor, and non-tumor cases.
4. **Evaluation**: Provides tools for assessing model performance and accuracy.

## Workflow

The project follows a structured workflow for analyzing brain MRI scans and detecting tumors. The main workflow is controlled by the `analysis_workflow()` function, which orchestrates the following steps:

1. **Dataset Loading**:
   - The workflow can either load the dataset from a source (e.g., "Mahadih534/brain-tumor-dataset") or from a previously saved local copy.
   - Controlled by `workflow_config['load_dataset']` and `workflow_config['load_dataset_from_disc']`.

2. **Data Exploration and Visualization**:
   - Display sample images from the dataset.
   - Explore dataset statistics and features.
   - Generate and save visualizations of tumor and non-tumor samples.
   - Controlled by various flags in `workflow_config`, such as `display_images`, `explore_dataset`, `get_all_labels`, etc.

3. **Model Application**:
   - Apply classification and/or detection models to the dataset.
   - The workflow supports both types of models, controlled by the `choose_model` dictionary.
   - Classification model: "Devarshi/Brain_Tumor_Classification"
   - Detection model: "DunnBC22/yolos-tiny-Brain_Tumor_Detection"

4. **Result Analysis and Visualization**:
   - Generate output images showing model predictions.
   - Save results in the `output_images` directory, organized by model type (classification/detection).

5. **Performance Comparison** (optional):
   - Compare execution times of models on CPU vs GPU.
   - Controlled by `workflow_config['compare_models']`.

### Workflow Configuration

The workflow is highly configurable through the `workflow_config` dictionary:

```python
workflow_config = {
    'load_dataset': False,
    'load_dataset_from_disc': True,
    'display_images': False,
    'display_and_save_images': False,
    'explore_dataset': False,
    'get_all_labels': False,
    'inspect_features': False,
    'get_tumor_samples': False,
    'apply_model': True,
    'compare_models': False
}
```

Users can easily enable or disable specific steps of the workflow by modifying these flags.

### Model Selection

The project allows for flexible model selection:

```python
choose_model = {
    'classification': True,
    'detection': True
}
```

By setting these flags, users can choose to run either the classification model, the detection model, or both.

This workflow design allows for easy experimentation with different datasets, models, and analysis steps, making the project adaptable to various research or clinical scenarios in brain tumor detection.

## Key Components

### Data Loading and Handling

The project provides flexible data loading options:

- `load_dataset_from_source(source: str, save_data_to_disc: bool = False)`: Loads dataset from a specified source with an option to save it to disk.
- `load_saved_dataset(dataset_dir: str)`: Loads a previously saved dataset from disk.

### Data Analysis and Exploration

Several functions are available for dataset exploration:

- `explore_dataset(dataset, dataset_name: str)`: Provides a comprehensive overview of the dataset, including sample count, basic statistics, and label distribution.
- `get_all_labels(data)`: Displays all unique labels in the dataset.
- `inspect_features(data)`: Shows available features and their first few entries.

### Image Processing and Visualization

The project includes functions for image handling and display:

- `display_images(data, num_images=5)`: Displays a specified number of images from the dataset.
- `display_and_save_images(data, num_images=5, output_dir='output_images')`: Displays and saves images to a specified directory.
- `get_tumor_samples(dataset, label: int, num_samples: int = 5, save_fig: bool = False, output_dir: str = 'output_images/tumor')`: Retrieves and optionally saves tumor or non-tumor samples.

### Model Application

The project supports both classification and detection models:

- `use_model(task: str, model: str, dataset, split_name: str = 'train', index: int = 0, field: str = 'image', save_image_dir=None, run_on_gpu: bool = True)`: Applies a specified model to an image from the dataset.
- `apply_object_detection_model(dataset, model_name, threshold=0.1, save_dir=None, split_name: str = 'train', index: int = 0, field: str = 'image', confidence_threshold: float = 0.8)`: Applies an object detection model to an image and visualizes the results.

### Performance Comparison

- `compare_execution_times(task, model, dataset, split_name, min_samples, max_samples, num_repeats=10)`: Compares execution times of the model on CPU and GPU.

## Model Comparison: Classification vs. Detection

### Visual Comparison

| Sample Index = 0 | Sample Index = 50 | Sample Index = 150 | Sample Index = 200 |
|-------------------|-------------------|---------------------|---------------------|
| ![Classification 0](Brain%20Tumor%20Analysis/output_images/model_analysis/object_classification/image_0.png) | ![Classification 50](Brain%20Tumor%20Analysis/output_images/model_analysis/object_classification/image_50.png) | ![Classification 150](Brain%20Tumor%20Analysis/output_images/model_analysis/object_classification/image_150.png) | ![Classification 200](Brain%20Tumor%20Analysis/output_images/model_analysis/object_classification/image_200.png) |
| ![Detection 0](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_0.png) | ![Detection 50](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_50.png) | ![Detection 150](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_150.png) | ![Detection 200](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_200.png) |

*Figure 1: Comparison of Classification (top row) and Detection (bottom row) model outputs for different sample indices*


#### Additional Detection Model Outputs

| Sample Index = 0 (variations) | Sample Index = 50 (variations) | Sample Index = 151 (variations) |
|-------------------------------|--------------------------------|--------------------------------|
| ![Detection 0_1](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_0_1.png) | ![Detection 50_1](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_50_1.png) | ![Detection 151](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_151.png) |
| ![Detection 0_2](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_0_2.png) | ![Detection 50_2](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_50_2.png) | ![Detection 151_1](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_151_1.png) |
| ![Detection 0_v1](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_0_v1.png) | ![Detection 50_3](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_50_3.png) | ![Detection 151_2](Brain%20Tumor%20Analysis/output_images/model_analysis/object_detection/image_151_2.png) |

*Figure 2: Additional outputs from the Detection model showing variations in tumor detection and localization*

## Usage

1. **Data Preparation:**
   Place your brain MRI dataset in the `datasets` directory or use the provided Mahadih534 brain tumor dataset.

2. **Load Dataset:**
   ```python
   dataset = load_dataset_from_source("Mahadih534/brain-tumor-dataset")
   # or
   dataset = load_saved_dataset("datasets/Mahadih534-brain-tumor-dataset")
   ```

3. **Explore Dataset:**
   ```python
   explore_dataset(dataset, "Mahadih534/brain-tumor-dataset")
   get_all_labels(dataset)
   inspect_features(dataset)
   ```

4. **Visualize Data:**
   ```python
   display_images(dataset, num_images=5)
   display_and_save_images(dataset, num_images=9, output_dir='output_images')
   ```

5. **Apply Models:**
   ```python
   use_model("image-classification", "Devarshi/Brain_Tumor_Classification", dataset)
   apply_object_detection_model(dataset, "DunnBC22/yolos-tiny-Brain_Tumor_Detection")
   ```

6. **Compare Performance:**
   ```python
   compare_execution_times("image-classification", "Devarshi/Brain_Tumor_Classification", dataset, 'train', 5, 10)
   ```

## Results

The `output_images` directory contains visual results from both classification and detection models. These images demonstrate the model's ability to identify and localize brain tumors in MRI scans.

### Non-Tumor Examples

![Non-tumor MRI example](Brain%20Tumor%20Analysis/output_images/non_tumor/image_1.png)
![Non-tumor MRI example](Brain%20Tumor%20Analysis/output_images/non_tumor/image_2.png)
![Non-tumor MRI example](Brain%20Tumor%20Analysis/output_images/non_tumor/image_3.png)

*Figure 3: Sample MRI scans classified as non-tumor*

### Tumor Examples

![Tumor MRI example](Brain%20Tumor%20Analysis/output_images/tumor/image_7.png)
![Tumor MRI example](Brain%20Tumor%20Analysis/output_images/tumor/image_4.png)
![Tumor MRI example](Brain%20Tumor%20Analysis/output_images/tumor/image_9.png)

*Figure 4: Sample MRI scans with detected tumors*

## Project Structure

```
.
├── datasets
│   └── Mahadih534-brain-tumor-dataset
├── output_images
│   ├── model_analysis
│   │   ├── object_classification
│   │   └── object_detection
│   ├── non_tumor
│   └── tumor
└── src
    ├── apply_models
    │   ├── evaluation.py
    │   ├── models.py
    │   └── models_util.py
    ├── utils
    │   ├── data_analysis.py
    │   ├── display_data.py
    │   └── util.py
    └── main.py
```
