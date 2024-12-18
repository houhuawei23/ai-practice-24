from torch.utils.data import Dataset, DataLoader
from PIL import Image


# Create Dataset Class
class ImageRegionDataset(Dataset):
    def __init__(self, annotations, category_to_idx, transform=None):
        self.annotations = annotations
        self.category_to_idx = category_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image = Image.open(annotation["path"]).convert("RGB")
        label = self.category_to_idx[annotation["category"]]

        if self.transform:
            image = self.transform(image)

        return image, label
