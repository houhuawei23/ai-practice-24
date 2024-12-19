from configs import get_image_json_output_paths
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2

from utils import load_SamAutoMaskGenerator, show_anns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

debug = True


def collect_segments_image(image, masks):
    test_segmentations = []
    for mask in masks:
        segmentation_mask = mask["segmentation"]
        # mask
        masked_image = np.zeros_like(image)  # 1024 x 1024 x 3
        masked_image[segmentation_mask] = image[segmentation_mask]
        # show
        # if test:
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(masked_image)
        #     print(masked_image.shape)
        #     break

        # crop
        bbox = mask["bbox"]
        x, y, w, h = map(
            int, [bbox[0], bbox[1], bbox[2], bbox[3]]
        )  # warning: x, y, w, h
        x_min, y_min, x_max, y_max = x, y, x + w, y + h
        cropped_image = masked_image[y_min:y_max, x_min:x_max]
        # print(
        #     f"bbox: {bbox}, x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}"
        # )
        # print(
        #     f"masked_image.shape: {masked_image.shape}, cropped_image.shape: {cropped_image.shape}"
        # )
        # resive
        img = Image.fromarray(cropped_image)
        # show
        # if debug:
        #     img.show()
        #     print(img.size)
        #     break
        # test_seg = transform(img).unsqueeze(0)
        test_segmentations.append(img)
    return test_segmentations


def main(image_id="3516"):
    image_path, json_path, output_dir = get_image_json_output_paths(image_id)
    print(image_path, json_path, output_dir)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # if debug:
    #     # cv2.imshow("image", image)
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(image)
    #     plt.show()
    # masks
    mask_generator = load_SamAutoMaskGenerator()
    masks = mask_generator.generate(image)
    # if debug:
    #     print(f"len(masks) = {len(masks)}")
    #     print(f"masks[0]")
    #     print(masks[0])
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(image)
    #     show_anns(masks)
    #     plt.axis("off")
    #     plt.show()

    # load model
    from utils import load_resnet18, load_label_colors, draw_predictions
    from configs import classifier_output_dir_path, categories, transform
    import os
    import torch

    model = load_resnet18(
        os.path.join(classifier_output_dir_path, "classifier.pth"), categories
    )

    # predict
    segments_images = collect_segments_image(image, masks)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    predicted_labels = []

    with torch.no_grad():
        for seg in segments_images:
            input = transform(seg).unsqueeze(0)
            output = model(input)
            _, predicted = torch.max(output, 1)
            predicted_labels.append(predicted.item())

    predicted_categories = [categories[label] for label in predicted_labels]
    category_to_color = load_label_colors()
    image_to_show = draw_predictions(
        image.copy(), masks, predicted_categories, category_to_color
    )
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(image_to_show)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
