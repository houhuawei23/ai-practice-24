import matplotlib.pyplot as plt
import cv2

from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms, models
from torch import nn
import torch
from PIL import Image

from train import parse_json_and_crop
import numpy as np

import yaml


# Function to draw bounding boxes with labels on the image
def draw_regions(
    image_path,
    annotations,
    category_to_idx,
    predicted_categories=None,
    category_to_color=None,
):
    """
    Draw bounding boxes and labels on the image.

    :param image_path: Path to the original image.
    :param annotations: List of ground truth or test annotations with 'bbox', 'category', 'segmentation' fields.
    :param category_to_idx: Dictionary mapping categories to indices.
    :param predicted_categories: List of predicted categories.
    :param category_to_color: Dictionary mapping categories to colors.

    :return: Image with bounding boxes and labels.
    """
    idx_to_category = {idx: category for category, idx in category_to_idx.items()}
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i, annotation in enumerate(annotations):
        bbox = annotation["bbox"]
        x_min, y_min, x_max, y_max = map(int, [bbox[0], bbox[1], bbox[2], bbox[3]])

        category = (
            annotation["category"]
            if predicted_categories is None
            else predicted_categories[i]
        )
        color = category_to_color.get(category, (255, 255, 255))

        segmentation = annotation["segmentation"]

        if segmentation:
            points = np.array(segmentation, dtype=np.int32).reshape((-1, 2))
            # Convert to Nx2 array
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

        cv2.putText(
            image,
            category,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return image


from utils import parse_json, load_label_colors


def visualize(
    image_path,
    json_path,
    categories,
    category_to_color,
    predicted_categories,
    test_accuracy=None,
    test_f1=None,
):
    annotations = parse_json(json_path)
    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    # Ground Truth Visualization
    ground_truth_image = draw_regions(
        image_path, annotations, category_to_idx, category_to_color=category_to_color
    )
    # Predictions Visualization
    predicted_image = draw_regions(
        image_path,
        annotations,
        category_to_idx,
        predicted_categories=predicted_categories,
        category_to_color=category_to_color,
    )
    # Plot the results
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth_image)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_image)
    plt.title("Predictions")
    plt.axis("off")

    plt.suptitle(f"Test on {image_path.split('/')[-1]}")
    if test_accuracy is not None and test_f1 is not None:
        res = f"Test Accuracy: {test_accuracy * 100:.2f}% | Test F1 Score: {test_f1:.2f}"
        plt.figtext(0.5, 0.01, res, wrap=True, horizontalalignment="center", fontsize=14)

    plt.show()


if __name__ == "__main__":
    import os
    from configs import (
        categories,
        category_to_idx,
        images_dir_path,
        cropped_output_dir_path,
        classifier_output_dir_path,
        transform,
        labels_path,
        get_image_json_output_paths,
    )
    from utils import load_resnet18, load_label_colors
    from test import test_classifier

    train_image_id = "3516"
    test_image_id = "3516"
    model_save_path = os.path.join(
        classifier_output_dir_path, f"classifier_trained_on_{train_image_id}.pth"
    )

    print(f"Testing classifier on {test_image_id}...")
    print(f"Loading model from {model_save_path}...")
    reloaded_model = load_resnet18(model_save_path, categories)
    test_image_id = test_image_id

    test_image_path, test_json_path, test_output_dir = get_image_json_output_paths(
        test_image_id
    )
    test_accuracy, test_f1, predicted_categories = test_classifier(
        reloaded_model,
        test_image_path=test_image_path,
        test_json_path=test_json_path,
        categories=categories,
        output_dir=test_output_dir,
        transform=transform,
    )
    print(f"train on {train_image_id}, test on {test_image_id}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test F1 Score: {test_f1:.2f}")

    category_to_color = load_label_colors(labels_path)

    visualize(
        test_image_path,
        test_json_path,
        categories,
        category_to_color,
        predicted_categories,
        test_accuracy=test_accuracy,
        test_f1=test_f1,
    )
