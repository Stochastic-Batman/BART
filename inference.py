import json
import joblib
import logging
import numpy as np
import os
import sys
import torch

from PIL import Image
from BART import BART


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger("Batlogger (Inference)")


def infer(path: str, model_path: str, is_dir: bool = True) -> None:
    label_encoder = joblib.load('label_encoder.joblib')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BART(num_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    results: dict[str, str] = {}

    if is_dir:
        files = [(f, os.path.join(path, f)) for f in sorted(os.listdir(path)) if f.lower().endswith('.jpg')]
    else:
        files = [(os.path.basename(path), path)]

    for f, p in files:
        with Image.open(p) as imag:
            img_array = np.array(imag.resize((128, 128)))

        img = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # During training DataLoader automatically creates batches to get shape (BS_size, C, H, W),
        # but during inference for a single image we only have (C, H, W), so we need to add batch dimension
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            class_idx = torch.max(model(img), 1)[1].item()

        results[f] = label_encoder.inverse_transform([class_idx])[0]

    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)

    logger.info(f"Saved predictions for {len(results)} images to results.json")


if __name__ == "__main__":
    b = len(sys.argv) > 1 and sys.argv[1].endswith('.jpg')
    infer(path=sys.argv[1] if b else 'inference_images', model_path='BART-10M.pth', is_dir=not b)