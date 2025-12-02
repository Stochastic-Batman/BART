import numpy as np
import os
from PIL import Image


def read_images(directory: str) -> tuple[list[np.ndarray], list[str], int]:
    images: list[np.ndarray] = []
    labels: list[str] = []
    class_count: int = 0

    for c in os.listdir(directory):
        if c in [".DS_Store", "simpsons_dataset"]:  # I think this subfolder should not be here...
            continue

        cp = os.path.join(directory, c)  # not that CP!
        class_count += 1

        for f in os.listdir(cp):
            fpath = os.path.join(cp, f)
            img = Image.open(fpath)
            assert(img.mode == 'RGB')  # I actually placed this here after I printed shapes, so it's kinda cheating
            img_array = np.array(img)
            images.append(img_array)
            labels.append(c)

    return images, labels, class_count


images, labels, class_count = read_images('characters_train')
print(f"The number of images ({len(images)}) is equal to the number of labels: {len(images) == len(labels)}")

img_shape_set: set[tuple[int, ...]] = set()
for img in images:
    img_shape_set.add(img.shape)

if len(img_shape_set) == 1:
    print(f"Dimensions of each image: {images[0].shape}")
else:
    print(f"There are {len(img_shape_set)} different sizes for images")
    print(f"Different shapes found: {sorted(img_shape_set)[::len(img_shape_set) - 1]}")  # Get only the smallest and the largest image shapes


# Looking at the data:
#     16,764 total samples (better than the last time, Shota)
#     275 different sizes, all RGB, most images have 256 pixels in one dimension

# There are 275 different sizes for images starting from (256, 256, 3) all the way to (1072, 1912, 3)
#
# ### 1. Resizing to 128 x 128
#
# Why resize:
# - 275 different image sizes make batching impossible without resizing. Unless you want to write some dumb out-of-this-world მატრაკვეცა case splits
# - Variable sizes would require padding/cropping, losing information unpredictably
#
# Why 128 x 128:
# - Most images have 256 pixels in one dimension - resizing to 128 x 128 means 2x downsampling, preserving general structure
# - Computational efficiency: 128 x 128 = 16,384 pixels vs 256 x 256 = 65,536 pixels (4x less memory/computation)
# - Sufficient for character recognition: Simpsons characters have distinctive colors and shapes that survive downsampling
# - Dataset size consideration: With only ~17K samples, smaller inputs reduce overfitting risk
#
# ### 2. BartSimpson Architecture
#
# Depth (4 conv blocks):
# - ~17K samples suggests moderate complexity - deeper networks would overfit, shallower might underfit
# - Rule of thumb: ~1000-5000 samples per conv block is reasonable
#
# Channel progression (32 -> 64 -> 128 -> 256):
# - Standard doubling pattern balances computational cost with feature e x traction
# - Starting at 32 (not 64) because dataset is modest-sized
# - Ending at 256 (not 512) to avoid overparameterization
#
# Dropout schedule (0.25 -> 0.25 -> 0.4 -> 0.4 -> 0.5):
# - Increases with depth where overfitting risk is higher
# - Conservative values (not 0.6-0.7) because we still need to learn features
#
# Two FC layers (512 -> 256 -> num_classes):
# - Single FC layer might be too simple for decision boundary
# - Three+ FC layers risk overfitting with this data amount
# - 512 and 256 units are proportional to final conv features (8 x 8 x 256 = 16,384)
#
# Total ~9M parameters:
# - Ratio of ~2 samples per parameter (17K samples / 9M params)
# - This is aggressive but manageable with proper regularization (dropout, data augmentation)
# - Modern practice shows this ratio can work with good augmentation
