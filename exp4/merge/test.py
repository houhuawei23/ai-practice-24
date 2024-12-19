from sklearn.metrics import accuracy_score, f1_score
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os

from utils import parse_json_and_crop


# Test the model
def test_classifier(
    model: torch.nn.Module,
    test_image_path: str,
    test_json_path: str,
    categories: list,
    output_dir: str,
    transform: torchvision.transforms.Compose,
):
    # Parse JSON and crop test regions
    annotations = parse_json_and_crop(test_image_path, test_json_path, output_dir)
    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    idx_to_category = {idx: category for category, idx in category_to_idx.items()}

    # Model evaluation mode
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    true_labels = []
    predicted_labels = []

    # Iterate over test annotations
    with torch.no_grad():
        for annotation in annotations:
            image = Image.open(annotation["path"]).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            true_label = category_to_idx[annotation["category"]]

            # Get model prediction
            output = model(image)
            _, predicted = torch.max(output, 1)

            true_labels.append(true_label)
            predicted_labels.append(predicted.item())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    # print(f"Accuracy: {accuracy * 100:.2f}%")
    # print(f"F1 Score: {f1:.2f}")
    predicted_categories = [idx_to_category[idx] for idx in predicted_labels]

    # ground truth categories count
    true_categories_count = {category: 0 for category in categories}
    for annotation in annotations:
        true_categories_count[annotation["category"]] += 1

    # predicted categories count
    predicted_categories_count = {category: 0 for category in categories}
    for category in predicted_categories:
        predicted_categories_count[category] += 1
    print(f"True Categories Count: {true_categories_count}")
    print(f"Predicted Categories Count: {predicted_categories_count}")
    return accuracy, f1, predicted_categories


if __name__ == "__main__":
    import os
    from torchvision import transforms, models
    from torch import nn

    from torchvision.models import resnet18, ResNet18_Weights

    imaage_name = "3532"
    test_image_path = f"./dataset/image/{imaage_name}.png"
    test_json_path = f"./dataset/image/{imaage_name}.json"
    test_cropped_regions = "./test_cropped_regions"
    output_dir = os.path.join(test_cropped_regions, imaage_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    from configs import categories, transform
    from utils import load_resnet18

    model = load_resnet18("classifier.pth", categories)

    # Assuming `model` is your trained model
    test_accuracy, test_f1 = test_classifier(
        model,
        test_image_path,
        test_json_path,
        categories,
        output_dir,
        transform=transform,
    )

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test F1 Score: {test_f1:.2f}")
