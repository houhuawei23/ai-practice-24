import os

categories = ["houses", "roads", "water", "trees", "ground"]
category_to_idx = {category: idx for idx, category in enumerate(categories)}

dataset_dir_path = os.path.join(os.getcwd(), "dataset")
images_dir_path = os.path.join(dataset_dir_path, "image")
newimages_dir_path = os.path.join(dataset_dir_path, "new_image")
cropped_output_dir_path = os.path.join(dataset_dir_path, "cropped")
classifier_output_dir_path = os.path.join(os.getcwd(), "classifiers")

labels_path = "./dataset/label/labels.yaml"

from torchvision import transforms

# Data transforms
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_image_json_output_paths(image_id: str, new_image: bool = False):
    if new_image:
        image_path = os.path.join(newimages_dir_path, f"{image_id}.png")
        json_path = os.path.join(newimages_dir_path, f"{image_id}.json")
    else:
        image_path = os.path.join(images_dir_path, f"{image_id}.png")
        json_path = os.path.join(images_dir_path, f"{image_id}.json")
    output_dir = os.path.join(cropped_output_dir_path, f"{image_id}")
    if not os.path.exists(image_path) or not os.path.exists(json_path):
        raise FileNotFoundError(f"Image or json file not found for {image_id}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return image_path, json_path, output_dir
