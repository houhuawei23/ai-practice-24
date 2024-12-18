from train import train_classifier
from test import test_classifier
import os
import torch
import argparse


def str_to_bool(value):
    if value.lower() in ["true", "1", "t", "y", "yes"]:
        return True
    elif value.lower() in ["false", "0", "f", "n", "no"]:
        return False
    else:
        raise argparse.ArgumentTypeError("bool value expected, got 'True' or 'False'")


parser = argparse.ArgumentParser()
parser.add_argument("--train_image_id", type=str, default="3516")
parser.add_argument("--test_image_id", type=str, default="3516")
parser.add_argument(
    "--train", type=str_to_bool, default=True, help="y or n, true or false"
)
parser.add_argument(
    "--test", type=str_to_bool, default=True, help="y or n, true or false"
)
parser.add_argument(
    "--visualize", type=str_to_bool, default=True, help="y or n, true or false"
)

args = parser.parse_args()

if __name__ == "__main__":
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

    train_image_id = args.train_image_id
    model_save_path = os.path.join(
        classifier_output_dir_path, f"classifier_trained_on_{train_image_id}.pth"
    )

    # train
    if args.train:
        print(f"Training classifier on {args.train_image_id} {args.train}...")

        train_image_path, train_json_path, train_output_dir = (
            get_image_json_output_paths(train_image_id)
        )

        model = train_classifier(
            train_image_path,
            train_json_path,
            train_output_dir,
            transform,
            categories,
        )
        torch.save(model.state_dict(), model_save_path)

    # test
    if args.test:
        print(f"Testing classifier on {args.test_image_id}...")
        print(f"Loading model from {model_save_path}...")
        reloaded_model = load_resnet18(model_save_path, categories)
        test_image_id = args.test_image_id
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

    if args.visualize:
        from visualize import visualize

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
