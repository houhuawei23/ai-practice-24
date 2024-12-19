import os
import cv2
import numpy as np
from PIL import Image


def create_new_dataset(source_dir, target_dir, mode="enlarge", factor=2):
    """
    Create a new dataset by enlarging or reducing images.

    :param source_dir: Directory containing the source images.
    :param target_dir: Directory to save the transformed images.
    :param mode: Mode of transformation ('enlarge' or 'reduce').
    :param factor: Enlargement or reduction factor (e.g., 2 for 4x, 3 for 9x).
    """
    os.makedirs(target_dir, exist_ok=True)
    image_files = [
        f
        for f in os.listdir(source_dir)
        if f.endswith((".png", ".jpg", ".jpeg", ".tif"))
    ]

    for image_file in image_files:
        image_path = os.path.join(source_dir, image_file)
        image = Image.open(image_path)
        image = image.convert("RGB")  # Ensure consistent mode

        if mode == "enlarge":
            new_images = enlarge_image(image, factor)
        elif mode == "reduce":
            new_images = reduce_image(image, factor)
        else:
            raise ValueError("Invalid mode. Use 'enlarge' or 'reduce'.")

        # Save new images to the target directory
        for i, new_image in enumerate(new_images):
            new_image_path = os.path.join(
                target_dir, f"{os.path.splitext(image_file)[0]}_{mode}_{i}.png"
            )
            new_image.save(new_image_path)

        print(f"Processed {image_file} -> {len(new_images)} new images.")


def enlarge_image(image, factor):
    """
    Enlarge an image by dividing it into smaller sections.

    :param image: PIL Image object.
    :param factor: Enlargement factor (e.g., 2 for 4 pieces, 3 for 9 pieces).
    :return: List of enlarged images (as PIL Images).
    """
    width, height = image.size
    new_images = []

    crop_width, crop_height = width // factor, height // factor

    for i in range(factor):
        for j in range(factor):
            left = j * crop_width
            upper = i * crop_height
            right = left + crop_width
            lower = upper + crop_height

            cropped_image = image.crop((left, upper, right, lower))
            resized_image = cropped_image.resize(
                (width, height), Image.Resampling.LANCZOS
            )
            new_images.append(resized_image)

    return new_images


def reduce_image(image, factor):
    """
    Reduce an image by creating scaled-down versions and resizing them back.

    :param image: PIL Image object.
    :param factor: Reduction factor (e.g., 2 for 4x smaller, 3 for 9x smaller).
    :return: List of reduced images (as PIL Images).
    """
    width, height = image.size
    new_images = []

    for _ in range(factor):
        small_image = image.resize(
            (width // factor, height // factor), Image.Resampling.LANCZOS
        )
        resized_image = small_image.resize((width, height), Image.Resampling.LANCZOS)
        new_images.append(resized_image)

    return new_images


# Example usage
source_dir = "./dataset/image"
target_dir = "./dataset/new_image"

# Enlarge dataset (split into 4 pieces and resize to original size)
create_new_dataset(source_dir, target_dir, mode="enlarge", factor=2)

# Reduce dataset (scale down by 4x and resize back to original size)
create_new_dataset(source_dir, target_dir, mode="reduce", factor=2)
