import json
import os

import cv2
import numpy as np

import yaml


def load_label_colors(labels_yaml_path):
    """
    Load category names and their corresponding colors from a YAML file.

    :param yaml_path: Path to the YAML file.
    :return: Dictionary mapping category names to colors (in RGB format).
    """
    with open(labels_yaml_path, "r") as file:
        labels = yaml.safe_load(file)["label"]

    # Map category names to colors (convert hex to RGB tuple)
    category_to_color = {
        label["name"]: tuple(
            int(label["color"].lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)
        )
        for label in labels
    }
    return category_to_color


def load_resnet18(model_path, categories):
    """
    Load the ResNet18 model from the given path.
    """
    import torch
    import torch.nn as nn
    from torchvision.models.resnet import resnet18, ResNet18_Weights

    # Load the model with updated arguments
    weights = ResNet18_Weights.DEFAULT  # Use default weights for ResNet18
    model = resnet18(weights=weights)  # Load ResNet18 with specific weights
    model.fc = nn.Linear(model.fc.in_features, len(categories))

    # Load model state dict
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return model


def parse_json(json_path):
    """
    Parse the JSON file and extract the category names and bounding boxes.

    """
    with open(json_path, "r") as file:
        data = json.load(file)
    annotations = []
    for obj in data["objects"]:
        category = obj["category"]
        bbox = obj["bbox"]
        x_min, y_min, x_max, y_max = map(int, [bbox[0], bbox[1], bbox[2], bbox[3]])
        segmentation = obj["segmentation"]
        # cropped_path = os.path.join(output_dir, f"{category}_{len(annotations)}.png")

        annotations.append(
            {
                # "path": cropped_path,
                "category": category,
                "bbox": [x_min, y_min, x_max, y_max],
                "segmentation": segmentation,
            }
        )
    return annotations


# Parse JSON and Crop Regions
def parse_json_and_crop(image_path, json_path, output_dir):
    """
    Parse the JSON file and crop the regions from the image.

    Args:
        image_path (str): Path to the image file.
        json_path (str): Path to the JSON file.
        output_dir (str): Path to the output directory.

    Returns:
        List[Dict]: List of dictionaries containing the cropped regions' information.
            path (str): Path to the cropped region's image file.
            category (str): Category of the object in the cropped region.
            bbox (List[int]): Bounding box of the cropped region.
            segmentation (List[[x, y]]): List of points that define the object's segmentation.
    """
    with open(json_path, "r") as file:
        data = json.load(file)

    image = cv2.imread(image_path)
    os.makedirs(output_dir, exist_ok=True)
    annotations = []

    for obj in data["objects"]:
        category = obj["category"]
        bbox = obj["bbox"]
        x_min, y_min, x_max, y_max = map(int, [bbox[0], bbox[1], bbox[2], bbox[3]])

        segmentation = obj["segmentation"]

        # Create a binary mask for the segmentation
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points = np.array(segmentation, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [points], color=255)

        # Mask the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        # Crop the image
        cropped = masked_image[y_min:y_max, x_min:x_max]
        cropped_path = os.path.join(output_dir, f"{category}_{len(annotations)}.png")
        cv2.imwrite(cropped_path, cropped)

        annotations.append(
            {
                "path": cropped_path,
                "category": category,
                "bbox": [x_min, y_min, x_max, y_max],
                "segmentation": segmentation,
            }
        )

    return annotations
