"""
from datasets import load_dataset
from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification
import timm
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np
from PIL import Image
import pandas as pd
"""

from src.apply_models.models import use_model, compare_execution_times, \
    apply_object_detection_model
from src.utils.data_analysis import explore_dataset, get_all_labels, inspect_features
from src.utils.display_data import display_images, display_and_save_images, get_tumor_samples
from src.utils.util import load_dataset_from_source, load_saved_dataset


def analysis_workflow():
    # Dataset path
    dataset_path = "Mahadih534/brain-tumor-dataset"

    output_dir = '../../output_images'

    # Models dict
    choose_model = {
        'classification': True,
        'detection': True
    }

    # Workflow control
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

    # Choose model
    classification_model, classification_task = "Devarshi/Brain_Tumor_Classification", "image-classification"
    detection_model, detection_task = "DunnBC22/yolos-tiny-Brain_Tumor_Detection", "object-detection"

    # Main workflow
    if workflow_config.get('load_dataset', True):
        dataset = load_dataset_from_source(source=dataset_path, save_data_to_disc=False)
        print(f"'{dataset_path}' dataset loaded successfully (source).")

    if workflow_config.get('load_dataset_from_disc', True):
        Mahadih534_dataset = "datasets/Mahadih534-brain-tumor-dataset"
        saved_data = Mahadih534_dataset
        dataset = load_saved_dataset(saved_data)
        print(f"'{saved_data}' dataset loaded successfully (disc).")

    if workflow_config.get('display_images', True) and workflow_config.get('load_dataset', True):
        print("\nDisplaying first set of images")
        num_images = 5
        display_images(dataset, num_images=num_images)

    if workflow_config.get('explore_dataset', True):
        print("\nExploring dataset with pandas:")
        explore_dataset(dataset, dataset_path)

    if workflow_config.get('get_all_labels', True):
        get_all_labels(dataset)

    if workflow_config.get('inspect_features', True):
        inspect_features(dataset)

    if workflow_config.get('display_and_save_images', True):
        num_images_to_save = 9
        display_and_save_images(dataset, num_images=num_images_to_save, output_dir=output_dir)

    if workflow_config.get('get_tumor_samples', True):
        save_fig = False
        label = 1  # The label number
        num_of_tumor_samples = 9
        tumor_output_dir = 'output_images/tumor'
        get_tumor_samples(dataset, label=label, num_samples=num_of_tumor_samples,
                          save_fig=save_fig, output_dir=tumor_output_dir)

    if workflow_config.get('apply_model', True):
        image_index = 1

        # Object classification model
        if choose_model.get('classification', True):
            save_image_dir = "output_images/model_analysis/object_classification"
            use_model(task=classification_task, model=classification_model, dataset=dataset, split_name='train', index=image_index,
                      field='image', save_image_dir=save_image_dir, run_on_gpu=False)
        # Object detection model
        if choose_model.get('detection', True):
            save_image_dir = 'output_images/model_analysis/object_detection'
            apply_object_detection_model(model_name=detection_model, dataset=dataset, split_name='train', index=image_index, field='image',
                                         threshold=None, save_dir=save_image_dir, confidence_threshold=0)

    if workflow_config.get('compare_models', True):
        compare_execution_times(task=classification_task, model=classification_model, dataset=dataset, split_name='train',
                                min_samples=5, max_samples=10)


def Three_Main_Stages():
    # Import dataset
    from datasets import load_dataset
    my_dataset = load_dataset("Mahadih534/brain-tumor-dataset")
    my_dataset.save_to_disk("path/to/data")  # Optional - save the data

    # Import model
    from transformers import pipeline
    classifier_or_detector = pipeline(task="image-classification",
                                      model="Devarshi/Brain_Tumor_Classification",
                                      device=0)  # 0 for GPU, -1 for CPU

    # Apply model
    image_sample = my_dataset["train"][0]["image"]
    result = classifier_or_detector(image_sample)
    print(result)  # Plot for visualization


def main():
    analysis_workflow()


if __name__ == '__main__':
    main()
