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


if __name__ == "__main__":
    from configs import transform

    image_path = "./dataset/image/3516.png"
    json_path = "./dataset/image/3516.json"
    output_dir = "cropped_regions"
    from configs import categories

    model = train_classifier(image_path, json_path, output_dir, transform, categories)

    torch.save(model.state_dict(), "classifier.pth")