import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

from utils import parse_json_and_crop
from dataset import ImageRegionDataset


def train_classifier(
    image_path,
    json_path,
    output_dir,
    transform,
    categories,
    category_to_idx=None,
    epochs=15,
    batch_size=32,
    lr=0.001,
    test_size=0.2,
):
    """
    Train a classifier on cropped regions of an image.
    Args:
        image_path (str): Path to the image file.
        json_path (str): Path to the JSON file containing the annotations.
        output_dir (str): Path to the directory where the cropped regions will be saved.
        categories (list): List of categories to classify.
        category_to_idx (dict): Dictionary mapping categories to indices.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for training.
        test_size (float): Fraction of data to use for testing.
    Returns:
        model (nn.Module): Trained classifier.
    """
    print("Training classifier...")
    # Parse JSON and crop regions
    annotations = parse_json_and_crop(image_path, json_path, output_dir)

    if category_to_idx is None:
        category_to_idx = {category: idx for idx, category in enumerate(categories)}

    # Split data into train and test
    train_annotations, test_annotations = train_test_split(
        annotations, test_size=test_size, random_state=42
    )

    # Datasets and Dataloaders
    train_dataset = ImageRegionDataset(
        train_annotations, category_to_idx, transform=transform
    )
    test_dataset = ImageRegionDataset(
        test_annotations, category_to_idx, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define CNN model (using a pretrained ResNet)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Update final layer for our categories
    model.fc = nn.Linear(model.fc.in_features, len(categories))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    # Evaluate on test data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total}%")
    return model


def batch_train_classifier(
    images_path_list,
    jsons_path_list,
    output_dir_list,
    transform,
    categories,
    category_to_idx=None,
    epochs=15,
    batch_size=32,
    lr=0.001,
    test_size=0.2,
    random_state=42,
):
    print("Training classifier on multiple images...")
    for image_path, json_path in zip(images_path_list, jsons_path_list):
        image_filename = image_path.split("/")[-1].split(".")[0]
        print(f"{image_filename}")

    annotations = []

    for image_path, json_path, output_dir in zip(
        images_path_list, jsons_path_list, output_dir_list
    ):
        cur_annotations = parse_json_and_crop(image_path, json_path, output_dir)
        annotations.extend(cur_annotations)

    if category_to_idx is None:
        category_to_idx = {category: idx for idx, category in enumerate(categories)}

    train_annotations, test_annotations = train_test_split(
        annotations, test_size=test_size, random_state=random_state
    )

    # Datasets and Dataloaders
    train_dataset = ImageRegionDataset(
        train_annotations, category_to_idx, transform=transform
    )
    test_dataset = ImageRegionDataset(
        test_annotations, category_to_idx, transform=transform
    )
    print(
        f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}"
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define CNN model (using a pretrained ResNet)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Update final layer for our categories
    model.fc = nn.Linear(model.fc.in_features, len(categories))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    # Evaluate on test data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total}%")
    return model


def test_train_classifier():
    from configs import transform

    image_path = "./dataset/image/3516.png"
    json_path = "./dataset/image/3516.json"
    output_dir = "cropped_regions"
    from configs import categories

    model = train_classifier(image_path, json_path, output_dir, transform, categories)

    torch.save(model.state_dict(), "classifier.pth")


import os


def test_batch_train_classifier():
    from configs import (
        transform,
        categories,
        dataset_dir_path,
        images_dir_path,
        get_image_json_output_paths,
        classifier_output_dir_path,
    )

    image_ids = ["3516", "3532", "3533"]
    images_path_list = []
    jsons_path_list = []
    output_dir_list = []
    for image_id in image_ids:
        image_path, json_path, output_dir = get_image_json_output_paths(image_id)
        images_path_list.append(image_path)
        jsons_path_list.append(json_path)
        output_dir_list.append(output_dir)
    model = batch_train_classifier(
        images_path_list,
        jsons_path_list,
        output_dir_list,
        transform,
        categories,
        epochs=5,
    )
    model_name = f"classifier_{'_'.join(image_ids)}.pth"
    torch.save(model.state_dict(), os.path.join(classifier_output_dir_path, model_name))


def test_batch_train_classifier_new():
    """
    Train RestNet18 on images with data augmentation.
    ./dataset/new_image: contains new images with different sizes and aspect ratios.

    """
    from configs import (
        transform,
        categories,
        dataset_dir_path,
        images_dir_path,
        get_image_json_output_paths,
        classifier_output_dir_path,
    )

    image_ids = ["3516_enlarge_0", "3532_enlarge_0", "3532_reduce_0"]
    images_path_list = []
    jsons_path_list = []
    output_dir_list = []
    for image_id in image_ids:
        image_path, json_path, output_dir = get_image_json_output_paths(
            image_id, new_image=True
        )
        if not os.path.exists(image_path) or not os.path.exists(json_path):
            print(f"{image_path} or {json_path} does not exist")
            continue

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        images_path_list.append(image_path)
        jsons_path_list.append(json_path)
        output_dir_list.append(output_dir)
    model = batch_train_classifier(
        images_path_list,
        jsons_path_list,
        output_dir_list,
        transform,
        categories,
        epochs=5,
    )
    model_name = f"classifier_{'_'.join(image_ids)}.pth"
    torch.save(model.state_dict(), os.path.join(classifier_output_dir_path, model_name))


if __name__ == "__main__":
    # test_batch_train_classifier()
    test_batch_train_classifier_new()
