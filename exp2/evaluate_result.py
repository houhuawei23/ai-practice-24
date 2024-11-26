import torch
import clip
from PIL import Image
import os
import pickle
import json
import time
import argparse

# python evaluate_result.py \
# --result_path="./dataset1/new.json" \
# --ground_truth_path="./dataset1/caption_with_keywords_and_image.json"

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, help='the path to store the image features', default='./result/caption_with_keywords_and_image.json')
parser.add_argument('--ground_truth_path', type=str, help='the path of the ground truth json file', default='./ground_truth/caption_with_keywords_and_image.json')
args = parser.parse_args()

with open(args.result_path, 'r') as file:
    pred = json.load(file)
with open(args.ground_truth_path, 'r') as file:
    gt = json.load(file)

caption_to_image = {}
for item in gt:
    caption_to_image[item['caption']] = item['image']
right_num = 0.0
for item in pred:
    if item['image'] == caption_to_image[item['caption']]:
        right_num += 1

with open('accuracy.txt', 'w') as file:
    file.write('accuracy:' + str(right_num/len(pred)*100) + "%")
print("the accuracy file has saved to accuracy.txt")
print('accuracy:', right_num/len(pred)*100, "%")