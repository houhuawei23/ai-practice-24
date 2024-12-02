import os
import cv2
import numpy as np
import torch
import clip
from PIL import Image
import pandas as pd

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Paths to data
template_path = "./LicensePlateData/Character_templates"
plate_path = "./LicensePlateData/License_plate"

# Load character templates


def load_templates(template_path):
    templates = []
    labels = []
    for file in os.listdir(template_path):
        if file.endswith(".jpg"):
            img_path = os.path.join(template_path, file)
            img = Image.open(img_path)
            img = preprocess(img).unsqueeze(0).to(device)

            templates.append(img)
            labels.append(os.path.splitext(file)[0])  # Use file name as label

    templates = torch.cat(templates, 0)

    # Encode character templates with CLIP
    with torch.no_grad():
        template_features = model.encode_image(templates)
    return templates, labels, template_features

def extract_characters_beta(image: Image.Image, num_chars=7, debug=False):
    x_range_list = [
        (10, 65),
        (65, 120),
        (145, 200),
        (205, 260),
        (260, 315),
        (315, 370),
        (375, 430),
    ]
    y_range = (20, 120)
    characters_images = []
    for x_range in x_range_list:
        x_min, x_max = x_range
        char_img = image.crop((x_min, y_range[0], x_max, y_range[1]))
        characters_images.append(char_img)

    if debug:
        image_with_boxes = np.array(image)
        # image_with_boxes = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        for i, char_img in enumerate(characters_images):
            x_min, x_max = x_range_list[i]
            cv2.rectangle(
                image_with_boxes,
                (x_min, y_range[0]),
                (x_max, y_range[1]),
                (0, 255, 0),
                2,
            )
        cv2.imshow("Char regions", image_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return characters_images


def extract_characters(image: np.ndarray, num_chars=7, debug=False):
    """Divide the image into character regions."""
    # h, w = image.shape
    x_range_list = [
        (10, 65),
        (65, 120),
        (145, 200),
        (205, 260),
        (260, 315),
        (315, 370),
        (375, 430),
    ]
    y_range = (20, 120)

    characters_images = []
    for x_range in x_range_list:
        x_min, x_max = x_range
        characters_images.append(image[y_range[0] : y_range[1], x_min:x_max])
    if debug:
        image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i, char_img in enumerate(characters_images):
            x_min, x_max = x_range_list[i]
            cv2.rectangle(
                image_with_boxes,
                (x_min, y_range[0]),
                (x_max, y_range[1]),
                (0, 255, 0),
                2,
            )
        cv2.imshow("Char regions", image_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return characters_images


def recognize_plate(plate_path, templates, labels, template_features):
    """Recognize license plate characters."""
    # print(f"Recognizing license plate in {plate_path}")
    # Read and preprocess the plate image
    plate_img = cv2.imread(plate_path, cv2.IMREAD_GRAYSCALE)

    # Extract characters
    char_regions = extract_characters(plate_img)

    plate_text = ""
    for char_img in char_regions:
        # Resize to match template size
        char_img = cv2.resize(char_img, (templates.shape[-1], templates.shape[-2]))
        char_tensor = preprocess(Image.fromarray(char_img)).unsqueeze(0).to(device)

        # Encode the character
        with torch.no_grad():
            char_features = model.encode_image(char_tensor)

        # Compare to template features
        # similarity = (char_features @ template_features.T).squeeze(0)
        # best_match = similarity.argmax().item()
        # print(f"template_features.shape: {template_features.shape}")
        # print(f"char_features.shape: {char_features.shape}")

        # normaize
        char_features = char_features / char_features.norm(dim=-1, keepdim=True)
        template_features = template_features / template_features.norm(
            dim=-1, keepdim=True
        )
        # template_features: torch.Size([65, 512]), char_features: torch.Size([1, 512])
        # Calculate similarity
        # TODO: get the best match
        similarity = torch.cosine_similarity(
            char_features.unsqueeze(1), template_features, dim=-1
        )

        best_match_idx = similarity.argmax().item()
        # print(f"similarity: {similarity}")
        # print(f"similarity.shape: {similarity.shape}")
        # print(f"best_match_idx: {best_match_idx}")

        # Append recognized character
        plate_text += labels[best_match_idx]

    return plate_text


def recognize_plate_beta(plate_path, templates, labels, template_features):
    """Recognize license plate characters."""
    # print(f"Recognizing license plate in {plate_path}")
    # Read and preprocess the plate image
    plate_img = Image.open(plate_path)
    # Extract characters
    char_regions = extract_characters_beta(plate_img, debug=True)

    plate_text = ""
    chars_features = []
    for char_img in char_regions:
        # Resize to match template size
        char_tensor = preprocess(char_img).unsqueeze(0).to(device)

        # Encode the character
        with torch.no_grad():
            char_features = model.encode_image(char_tensor)
        chars_features.append(char_features)

    chars_features = torch.cat(chars_features, 0)

    # Compare to template features
    chars_features = chars_features / chars_features.norm(dim=-1, keepdim=True)
    template_features = template_features / template_features.norm(
        dim=-1, keepdim=True
    )
    similarity = torch.cosine_similarity(
        chars_features.unsqueeze(1), template_features, dim=-1
    )
    # print(f"similarity.shape: {similarity.shape}")
    # similarity.shape: torch.Size([7, 65])
    best_match_indices = similarity.argmax(dim=1)
    # print(f"best_match_indices: {best_match_indices}")
    for i in range(similarity.shape[0]):
        best_match_idx = best_match_indices[i].item()
        # print(f"best_match_idx: {best_match_idx}, char: {labels[best_match_idx]}")
        plate_text += labels[best_match_idx]



    # normaize
    # char_features = char_features / char_features.norm(dim=-1, keepdim=True)
    # template_features = template_features / template_features.norm(
    #     dim=-1, keepdim=True
    # )

    # similarity = torch.cosine_similarity(
    #     char_features.unsqueeze(1), template_features, dim=-1
    # )

    # best_match_idx = similarity.argmax().item()

    # # Append recognized character
    # plate_text += labels[best_match_idx]

    return plate_text



if __name__ == "__main__":
    templates, labels, template_features = load_templates(template_path)

    # Process all plates and save results
    # calculate accuracy
    results = []
    correct_cnt = 0
    for file in os.listdir(plate_path):
        if file.endswith(".jpg"):
            true_number = file.split("_")[0]

            full_path = os.path.join(plate_path, file)
            plate_number = recognize_plate_beta(
                full_path, templates, labels, template_features
            )
            if plate_number == true_number:
                print(f"Correct: {true_number} == {plate_number}")
                correct_cnt += 1
            else:
                print(f"Incorrect: {true_number} != {plate_number}")
                
            # print(f"Plate number for {file} is {plate_number}")
            results.append({"Filename": file, "LicensePlate": plate_number})
    
    accuracy = correct_cnt / len(results)
    print(f"Accuracy: {accuracy}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print("Recognition complete. Results saved to 'results.csv'.")
