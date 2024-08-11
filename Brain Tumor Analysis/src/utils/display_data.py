import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
from src.utils.util import *


def display_and_save_images(data, num_images=5, output_dir='output_images'):
    # Check if the dataset has a 'train' split
    check_train_split(data)

    num_samples = min(num_images, len(data['train']))
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))

    # Define the label mapping dictionary
    label_map = {0: 'No', 1: 'Yes'}

    num_of_images = 0

    for i in range(num_samples):
        output_path = f'{output_dir}/image_{i + 1}.png'
        image = data['train'][i]['image']

        # Try to get label information
        label = "Unknown"
        if 'label' in data['train'][i]:
            original_label = data['train'][i]['label']
            label = label_map.get(original_label, str(original_label))
        elif 'filename' in data['train'][i]:
            label = data['train'][i]['filename']

        # Save individual image
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Image {i + 1}\nLabel: {label}")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        num_of_images += 1

        # Display in the main figure
        if isinstance(axes, np.ndarray):
            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(f"Image {i + 1}\nLabel: {label}")
        else:
            axes.imshow(image)
            axes.axis('off')
            axes.set_title(f"Image {i + 1}\nLabel: {label}")

    plt.tight_layout()
    plt.show()

    print(f"{i + 1} images saved in the '{output_dir}' folder.")


def display_images(data, num_images=5):
    # Check if the dataset has a 'train' split
    check_train_split(data)

    num_samples = min(num_images, len(data['train']))
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))

    # Define the label mapping dictionary
    label_map = {0: 'No', 1: 'Yes'}

    for i in range(num_samples):
        image = data['train'][i]['image']
        if isinstance(axes, np.ndarray):
            axes[i].imshow(image)
            axes[i].axis('off')

            # Try to get label information
            label = "Unknown"
            if 'label' in data['train'][i]:
                original_label = data['train'][i]['label']
                label = label_map.get(original_label, str(original_label))
            elif 'filename' in data['train'][i]:
                label = data['train'][i]['filename']

            axes[i].set_title(f"Image {i + 1}\nLabel: {label}")
        else:
            axes.imshow(image)
            axes.axis('off')
            axes.set_title(f"Image {i + 1}\nLabel: {label}")

    plt.tight_layout()
    plt.show()


def get_tumor_samples(dataset, label: int, num_samples: int = 5, save_fig: bool = False,
                      output_dir: str = 'output_images/tumor'):
    # Check if the dataset has a 'train' split
    check_train_split(dataset)

    train_data = dataset['train']

    if label == 0:
        # Filter for samples with label 0 (non-tumor)
        filtered_dataset = dataset.filter(non_tumor_filter_func)
    elif label == 1:
        # Filter for samples with label 1 (tumor)
        filtered_dataset = dataset.filter(tumor_filter_func)
    else:
        print("Invalid label. Please provide a label of either 0 or 1.")
        return

    if save_fig:
        display_and_save_images(filtered_dataset, num_images=num_samples, output_dir=output_dir)
        return
    else:
        display_images(dataset, num_images=num_samples)


def display_image_from_dataset(label, image, index, save_dir=None):
    print(f'Image_{index}: [Label: {label}] [{type(image)}]')

    plt.figure()
    plt.title(f'Label: {label}\nIndex: {index}')
    plt.imshow(image)
    plt.axis('off')

    if save_dir is not None:
        save_path = f'{save_dir}/image_{index}.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Image saved to {save_path}')

    plt.show()
