import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset


def get_data(directory: str) -> tuple[list[Image.Image], list[str], int]:
    images: list[Image.Image] = []
    labels: list[str] = []
    class_count: int = 0

    for c in sorted(os.listdir(directory)):
        if c in [".DS_Store", "simpsons_dataset"]:  # Remnants of the past...
            continue

        cp = os.path.join(directory, c)  # not that CP!
        class_count += 1

        for f in sorted(os.listdir(cp)):
            fpath = os.path.join(cp, f)
            with Image.open(fpath) as img:
                assert(img.mode == 'RGB')  # I actually placed this here after I printed shapes, so it's kinda cheating
                images.append(img.copy())  # .copy() loads image into memor, cause PIL loads images lazily
            labels.append(c)

    return images, labels, class_count


class SimpsonsDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels


    def __len__(self) -> int:
        return len(self.images)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]

        # convert to tensor and normalize, change from (H, W, C) to (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(label, dtype=torch.long)

        return image, label