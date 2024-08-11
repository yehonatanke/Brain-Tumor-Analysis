from datasets import load_dataset, load_from_disk
import timm
import torch


def load_dataset_from_source(source: str, save_data_to_disc: bool = False):
    """
    Loads the dataset from the specified source.

    Args:
        source (str): The source from which to load the dataset.
        save_data_to_disc (bool): If save the data

    Returns:
        Dataset: The loaded dataset if successfully loaded.
        None: If the dataset fails to load or is empty.
    """
    try:
        dataset = load_dataset(source)

        if not dataset:
            print("Error: The dataset is empty or failed to load.")
            return None
        if save_data_to_disc:
            # Set dataset saving name
            dataset_name = source.replace('/', '-')
            save_path = f"datasets/{dataset_name}"
            dataset.save_to_disk(save_path)
            print(f"Dataset '{source}' successfully saved to '{save_path}'.")

        return dataset

    except Exception as e:
        print(f"An error occurred while loading the dataset from {source}: {e}")
        return None


def load_saved_dataset(dataset_dir: str):
    try:
        # Load the dataset from disk
        dataset = load_from_disk(dataset_dir)
        print(f"Dataset successfully loaded from '{dataset_dir}'.")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset from '{dataset_dir}': {e}")
        return None


def check_train_split(dataset):
    # Check if the dataset has a 'train' split
    if 'train' not in dataset:
        raise ValueError("The dataset does not have a 'train' split.")
    return dataset


def check_dataset_fields(dataset, split_name: str, index: int, field: str):
    # Check if the split exists
    if split_name not in dataset:
        print(f"Split '{split_name}' not found in the dataset.")
        return False

    # Check if the index is within the valid range
    if index < 0 or index >= len(dataset[split_name]):
        print(f"Index {index} is out of range for split '{split_name}'.")
        return False

    # Check if the specified field exists in the dataset
    if field not in dataset[split_name].column_names:
        print(f"'{field}' field not found in the dataset.")
        return False

    return True


# Define a filter function
def tumor_filter_func(sample):
    return sample['label'] == 1


def non_tumor_filter_func(sample):
    return sample['label'] == 0
