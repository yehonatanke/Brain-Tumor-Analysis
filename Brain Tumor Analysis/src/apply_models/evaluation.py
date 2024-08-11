import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

def evaluate_model_on_custom_dataset(model, dataset, batch_size=32, num_classes=None, device='cuda'):
    # Move model to the specified device
    model = model.to(device)
    model.eval()

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Lists to store true labels and predictions
    all_labels = []
    all_preds = []

    # Disable gradient calculations
    with torch.no_grad():
        for images, labels in dataloader:
            # Move data to the same device as the model
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Store the results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Infer number of classes if not provided
    if num_classes is None:
        num_classes = len(np.unique(all_labels))

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate per-class metrics
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # Calculate misses (1 - recall)
    misses = 1 - recall
    per_class_misses = 1 - per_class_recall

    # Prepare the results dictionary
    results = {
        'overall': {
            'accuracy': {'percentage': accuracy * 100, 'nominal': int(accuracy * len(all_labels))},
            'precision': {'percentage': precision * 100, 'nominal': int(precision * len(all_labels))},
            'recall': {'percentage': recall * 100, 'nominal': int(recall * len(all_labels))},
            'f1_score': {'percentage': f1 * 100, 'nominal': None},
            'misses': {'percentage': misses * 100, 'nominal': int(misses * len(all_labels))}
        },
        'per_class': {
            f'class_{i}': {
                'precision': {'percentage': per_class_precision[i] * 100, 'nominal': int(per_class_precision[i] * np.sum(all_labels == i))},
                'recall': {'percentage': per_class_recall[i] * 100, 'nominal': int(per_class_recall[i] * np.sum(all_labels == i))},
                'f1_score': {'percentage': per_class_f1[i] * 100, 'nominal': None},
                'misses': {'percentage': per_class_misses[i] * 100, 'nominal': int(per_class_misses[i] * np.sum(all_labels == i))}
            } for i in range(num_classes)
        },
        'confusion_matrix': cm.tolist()
    }

    return results