Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline
实现端到端车牌检测和识别：大型数据集和基线

Zhenbo Xu, Wei Yang, Ajin Meng, Nanxue Lu, Huan Huang, Changchun Ying, Liusheng Huang; Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 255-271
徐振波，杨伟，孟阿金，路南学，黄焕，应长春，黄柳生；欧洲计算机视觉会议 (ECCV) 会议记录，2018 年，第 255-271 页

Abstract

Most current license plate (LP) detection and recognition approaches are evaluated on a small and usually unrepresentative dataset since there are no publicly available large diverse datasets. In this paper, we introduce CCPD, a large and comprehensive LP dataset. All images are taken manually by workers of a roadside parking management company and are annotated carefully. To our best knowledge, CCPD is the largest publicly available LP dataset to date with over 250k unique car images, and the only one provides vertices location annotations. With CCPD, we present a novel network model which can predict the bounding box and recognize the corresponding LP number simultaneously with high speed and accuracy. Through comparative experiments, we demonstrate our model outperforms current object detection and recognition approaches in both accuracy and speed. In real-world applications, our model recognizes LP numbers directly from relatively high-resolution images at over 61 fps and 98.5% accuracy.

由于没有公开可用的大型多样化数据集，当前大多数车牌（LP）检测和识别方法都是在小型且通常不具代表性的数据集上进行评估。在本文中，我们介绍了 CCPD，一个大型且全面的 LP 数据集。所有图像均由路边停车管理公司的工作人员手动拍摄，并经过仔细注释。据我们所知，CCPD 是迄今为止最大的公开可用的 LP 数据集，拥有超过 25 万张独特的汽车图像，并且是唯一提供顶点位置注释的数据集。通过 CCPD，我们提出了一种新颖的网络模型，可以高速、准确地预测边界框并同时识别相应的 LP 数。通过比较实验，我们证明我们的模型在准确性和速度上都优于当前的对象检测和识别方法。在实际应用中，我们的模型以超过 61 fps 的速度和 98.5% 的准确度直接从相对高分辨率的图像中识别 LP 编号。

# ChatGPT

License Plate Recognition:

- Using Python, include numpy, pandas, matplotlib, opencv-python, clip, PIL, torch and so on.
- DataSet: Two folders are provided, `License_plate` stores the license plates to be recognized, and `Character_templates` stores the character image templates that have appeared in all license plates.
- Since the size of all license plate pictures in the data set is the same, please:
  - first divide the license plate into character areas,
  - and then match it with the character picture template,
  - and the matching method uses the feature calculation similarity obtained by the clip model,
  - and finally obtain the license plate result.
- The experiment required to be able to read the picture of the license plate and output the license plate number.

```bash
./LicensePlateDataTest
├── Character_templates
│   ├── 0.jpg
│   ├── 1.jpg
│   └── 2.jpg
|     ....
|     ....
└── License_plate
    ├── ahy1N6L77_blue_False.jpg
    ├── ahy4SD8V2_blue_False.jpg
    └── ahy5LBR4E_blue_False.jpg
    |   ....
    |   ....
```

CLIP usage example:

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```
