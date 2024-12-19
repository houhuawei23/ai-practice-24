import json
import os

import cv2
import numpy as np

import yaml

from typing import List, Dict, Tuple


def draw_predictions(
    image_to_show: np.ndarray,
    masks: List[Dict],
    predicted_categories: List[str],
    category_to_color: Dict[str, Tuple[int, int, int]],
):
    for mask, predicted_category in zip(masks, predicted_categories):
        color = category_to_color[predicted_category]
        segmentation = mask[
            "segmentation"
        ]  # ndarray, type = bool, shape = (height, width)
        x, y, w, h = mask["bbox"]  # x, y: top-left corner, w, h: width and height
        # print(f"category: {predicted_category}, color: {color}, bbox: {x}, {y}, {w}, {h}")
        # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        # TODO: draw the outline of the segmenation mask
        # Find contours from the segmentation mask
        contours, _ = cv2.findContours(
            segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Draw the contours on the image
        cv2.drawContours(image_to_show, contours, -1, color, 2)
        # category label
        cv2.putText(
            image_to_show,
            predicted_category,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return image_to_show


def show_anns(anns: List[Dict]):
    """
    anns: List[Dict]
        "segmentation":
            ndarray, type=bool, shape=(H, W), values=True where the segment is present
        "area": int, area of the segment
        "bbox": List[int], [x, y, w, h] of the segment
        "predicted_iou": float, IOU of the segment with the predicted mask
        "point_coords": List[List[int]], list of points that define the segment
        "stability_score": float, stability score of the segment
        "crop_box": List[int], [x, y, w, h] of the crop box used to generate the segment
    """
    import matplotlib.pyplot as plt

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def load_SamAutoMaskGenerator(
    sam_checkpoint="../reps/sam_vit_b_01ec64.pth", model_type="vit_b", device="cpu"
):
    """
    Load the SamAutoMaskGenerator model from the given path.
    """
    import torch
    from configs import get_image_json_output_paths
    from segment_anything import (
        sam_model_registry,
        SamAutomaticMaskGenerator,
        SamPredictor,
    )

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def load_label_colors(labels_yaml_path="./dataset/label/labels.yaml"):
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


from configs import categories


def load_resnet18(model_path, categories=categories):
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
