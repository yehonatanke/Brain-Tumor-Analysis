from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import time
import random

import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, pipeline
from PIL import Image, ImageDraw
import numpy as np

from src.apply_models.models_util import get_image_and_label, save_image, GREEN, BLUE, YELLOW, RED, RESET
from src.utils.display_data import display_image_from_dataset
from src.utils.util import check_dataset_fields


def use_model(task: str, model: str, dataset, split_name: str = 'train', index: int = 0,
              field: str = 'image', save_image_dir=None, run_on_gpu: bool = True):
    image, label = get_image_and_label(dataset=dataset, split_name=split_name, index=index, field=field)
    # Check validation
    if image is None or label is None:
        return

    # Display the image
    try:
        display_image_from_dataset(label, image, index, save_dir=save_image_dir)
    except Exception as e:
        print(f"An error occurred while opening the image: {e}")

    # Import model
    if run_on_gpu:
        classifier = pipeline(task, model=model, device=0)
    else:
        classifier = pipeline(task, model=model)

    # Convert image to RGB if it's not
    if image.mode != "RGB":
        image = image.convert("RGB")
        print("Image converted to RGB")

    # Apply model
    result = classifier(image)

    # Show result
    print(f"---\nResult for image_{index}:\n{result}\nEnd Result\n---")


def apply_object_detection_model(dataset, model_name, threshold=0.1, save_dir=None,
                                 split_name: str = 'train', index: int = 0, field: str = 'image', confidence_threshold: float = 0.8):
    print(f"{RED}---\nObject Decection:\nModel: {model_name}{RESET}")
    print(f"{RED}Confidence threshold: {threshold}{RESET}")

    # Load model
    try:
        pipe = pipeline("object-detection", model=model_name)
        print(f"{YELLOW}Model loaded successfully{RESET}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load and process image
    try:
        # image = Image.open(image_path)
        image, label = get_image_and_label(dataset=dataset, split_name=split_name, index=index, field=field)
        print(f"{YELLOW}Image loaded successfully. Size: {image.size}, Mode: {image.mode}{RESET}")

        # Convert image to RGB if it's not
        if image.mode != "RGB":
            image = image.convert("RGB")
            print("Image converted to RGB")

        # Run inference
        if threshold is not None:
            result = pipe(image, threshold=threshold)
        else:
            result = pipe(image)
        print(f"{YELLOW}Inference completed. {len(result)} objects detected.")

        labels = pipe.model.config.id2label.values()
        print(f"{BLUE}Labels: {labels}{RESET}")

        # Print results
        for detection in result:
            print(
                f"{RED}[Detected]: ['{detection['label']}'] with confidence {detection['score']:.4f} at location {detection['box']}{RESET}")

        # Visualize results
        draw = ImageDraw.Draw(image)

        for detection in result:
            if detection['score'] > confidence_threshold:
                box = detection['box']
                draw.rectangle(((box['xmin'], box['ymin']), (box['xmax'], box['ymax'])), outline="red", width=1)
                draw.text((box['xmin'], box['ymin']), f"{detection['label']}\n{detection['score']:.2f}", fill="white")

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title(f'Label: {label}, Index: {index}\nConfidence Threshold: {confidence_threshold}, Threshold: {threshold}')
        plt.axis('off')

        if save_dir is not None:
            save_image(save_dir, index, plt)

        plt.show()

    except Exception as e:
        print(f"Error processing image or running inference: {e}")
        return

    print("Process [Object Detection] Complete\n---")


def compare_execution_times(task, model, dataset, split_name, min_samples, max_samples, num_repeats=10):
    # Initialize the pipelines for CPU and GPU
    classifier_cpu = pipeline(task, model=model, device=-1)
    classifier_gpu = pipeline(task, model=model, device=0)

    # Measure execution time on CPU
    cpu_times = []
    for _ in range(num_repeats):
        images = select_random_images(dataset, split_name, min_samples, max_samples)
        start_time = time.time()
        for image in images:
            classifier_cpu(image)
        cpu_times.append(time.time() - start_time)

    # Measure execution time on GPU
    gpu_times = []
    for _ in range(num_repeats):
        images = select_random_images(dataset, split_name, min_samples, max_samples)
        start_time = time.time()
        for image in images:
            classifier_gpu(image)
        gpu_times.append(time.time() - start_time)

    # Compute average times
    avg_cpu_time = sum(cpu_times) / num_repeats
    avg_gpu_time = sum(gpu_times) / num_repeats

    print(f"Average CPU time: {avg_cpu_time:.4f} seconds")
    print(f"Average GPU time: {avg_gpu_time:.4f} seconds")

    # Determine which device is faster (nominal)
    if avg_cpu_time < avg_gpu_time:
        faster_device = "CPU"
        time_difference = avg_gpu_time - avg_cpu_time
    else:
        faster_device = "GPU"
        time_difference = avg_cpu_time - avg_gpu_time

    print(f"The {faster_device} is faster by {time_difference:.4f} seconds")

    # Determine which device is faster (%)
    if avg_cpu_time < avg_gpu_time:
        faster_device = "CPU"
        percentage_faster = ((avg_gpu_time - avg_cpu_time) / avg_gpu_time) * 100
    else:
        faster_device = "GPU"
        percentage_faster = ((avg_cpu_time - avg_gpu_time) / avg_cpu_time) * 100

    print(f"The {faster_device} is faster by {percentage_faster:.2f}%")


def select_random_images(dataset, split_name, min_samples, max_samples):
    # Ensure the number of samples is within the dataset range
    num_samples = random.randint(min_samples, max_samples)
    max_index = len(dataset[split_name]) - 1

    # Randomly select indices
    indices = random.sample(range(max_index + 1), num_samples)

    # Retrieve images based on selected indices
    images = [dataset[split_name][i]['image'] for i in indices]

    return images
