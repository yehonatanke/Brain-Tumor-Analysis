import numpy as np
from PIL import Image
import io
import pandas as pd


def explore_dataset(dataset, dataset_name: str):
    # Check if the dataset has a 'train' split
    if 'train' not in dataset:
        raise ValueError("The dataset does not have a 'train' split.")

    # Get the train split
    train_data = dataset['train']

    # Convert to pandas DataFrame
    df = pd.DataFrame(train_data)

    print(f"Dataset: {dataset_name}")
    print(f"Number of samples: {len(df)}")

    # Display the first few rows of the dataset
    print("\nFirst few rows of the dataset:")
    print(df.head())

    # Display dataset information
    print("\nDataset info:")
    print(df.info())

    # Display basic statistics
    print("\nBasic statistics:")
    print(df.describe(include='all'))

    # Check if 'label' exists in the dataset
    if 'label' in df.columns:
        print("\nLabel distribution:")
        print(df['label'].value_counts())

    # Image statistics
    if 'image' in df.columns:
        # Function to get image shape and mean pixel value
        def image_stats(img):
            pil_img = Image.open(io.BytesIO(img['bytes']))
            np_img = np.array(pil_img)
            return pd.Series({
                'height': np_img.shape[0],
                'width': np_img.shape[1],
                'channels': np_img.shape[2] if len(np_img.shape) > 2 else 1,
                'mean_pixel_value': np_img.mean()
            })


def get_all_labels(data):
    if 'label' in data['train'].features:
        labels = data['train']['label']
        unique_labels = set(labels)
        print("\nAll unique labels in the dataset:")
        for label in sorted(unique_labels):
            print(f"- {label}")
        print(f"\nTotal number of unique labels: {len(unique_labels)}")
    else:
        print("\nNo 'label' feature found in the dataset.")


def inspect_features(data):
    print("\nAvailable features in the dataset:")
    for feature, feature_type in data['train'].features.items():
        print(f"- {feature}: {feature_type}")

    print("\nFirst few entries of each feature:")
    for feature in data['train'].features:
        print(f"\n{feature}:")
        print(data['train'][feature][:5])  # Print first 5 entries
