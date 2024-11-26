from typing import List, Dict, Tuple, Callable
import json

from PIL import Image
import torch
from clip.model import CLIP

from keywords_extract import extract_keywords_tfidf


def choose_best_image_for_caption(
    caption: str,
    images: List[Tuple[str, Image.Image]],
    model: CLIP | torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
):
    """ """
    # pass
    images_preprocessed = torch.stack([preprocess(image[1]) for image in images])
    # keywords =
    with torch.no_grad():
        image_features = model.encode_image(images_preprocessed)


def read_caption_file(file_path: str = "./dataset1/caption.json", debug=False):
    # read the caption
    with open(file_path, "r") as f:
        captions = json.load(f)

    caption_list = [item["caption"] for item in captions]

    if debug:
        print("size of caption list: ", len(caption_list))
        print("caption and caption_list example: ")
        print(captions[0])
        print(caption_list[0])

    # return captions, caption_list # Tuple[List[Dict], List[str]]
    infos = dict()
    cnt = 0
    for caption in captions:
        infos[cnt] = {"caption": caption["caption"]}
        cnt += 1

    return infos


import clip


# use our keywords extractor
from keywords_extract import extract_keywords_tfidf
import os

# keywords_list = keyWords_extractor(caption_list)

import numpy as np

# info: Dict[caption: str -> ]
# infos: Dict[id -> info]


def predict_best_image(
    # keywords_list,
    infos: Dict[int, Dict],
    images_path="./dataset1/images/",
    debug=False,
):
    # check infos[idx] must have "caption" and "keywords" keys
    for info in infos.values():
        if "caption" not in info or "keywords" not in info:
            print("Error: info must have 'caption' and 'keywords' keys")
            return
    # idx = 0
    # if infos.get(idx) and ("caption" not in infos[idx] or "keywords" not in infos[idx]):
    #     print("Error: infos[idx] must have 'caption' and 'keywords' keys")
    #     return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)

    keywords_list = [info["keywords"] for info in infos.values()]

    captions_tensor = clip.tokenize(keywords_list).to(device)

    # Load the images and preprocess them
    images_files = os.listdir(images_path)

    images = [
        Image.open(images_path + f).convert("RGB") for f in os.listdir(images_path)
    ]

    images_preprocessed = torch.stack([preprocess(img) for img in images]).to(device)

    # Calculate the similarity between the images and the captions

    with torch.no_grad():
        image_features = model.encode_image(images_preprocessed)
        text_features = model.encode_text(captions_tensor)
        logits_per_image, logits_per_text = model(images_preprocessed, captions_tensor)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    if debug:
        print("Probs: ")
        print(probs)

    # select the best image for each caption
    best_image_indices = np.argmax(probs, axis=0)
    # print image paths for the best images
    best_image_paths = [images_files[i] for i in best_image_indices]

    if debug:
        for i in range(min(len(keywords_list), 2)):
            print(f"Caption: {keywords_list[i]}")
            print(f"Best image: {best_image_paths[i]}")
            print("-")

    # update infos with best image paths
    for key, info in infos.items():
        info["image"] = best_image_paths[key]

    if debug:
        list_infos = list(infos.values())
        for info in list_infos[: min(len(list_infos), 2)]:
            print(f"Caption: {info['caption']}")
            print(f"Keywords: {info['keywords']}")
            print(f"Best image: {info['image']}")
            print("-")


def save_result(
    infos: Dict[int, Dict], output_path: str = "./result.json", debug=False
):
    # save to result.json
    result = []
    for info in infos.values():
        result.append(info)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    if debug:
        print(f"Result saved to {output_path}")


from keywords_extract import extract_keywords

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    type=str,
    default="gt",
    help="keywords extraction method: gt|tfidf|cutoff|keybert",
)

args = parser.parse_args()

import os


def main():
    debug = True
    # read the caption file
    keywords_num = 2
    infos = read_caption_file("./dataset1/caption.json", debug=debug)
    extract_keywords(infos, method=args.method, keywords_num=keywords_num, debug=debug)
    predict_best_image(infos, images_path="./dataset1/images/", debug=debug)
    
    results_dir = "./results"
    # output_path = f"./result_{args.method}_{keywords_num}.json"'
    output_path = os.path.join(results_dir, f"result_{args.method}_{keywords_num}.json")
    save_result(infos, output_path=output_path, debug=debug)

    # run the evaluate script:
    # python evaluate_result.py --result_path ./result.json --ground_truth_path ./dataset1/caption.json
    import subprocess

    evaluate_script_path = "./evaluate_result.py"
    gt_path = "./dataset1/caption_with_keywords_and_image.json"
    cmd = f"python {evaluate_script_path} --result_path {output_path} --ground_truth_path {gt_path}"
    subprocess.run(cmd, shell=True)
    print("Evaluation done.")


if __name__ == "__main__":
    main()
