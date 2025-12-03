import argparse
import joblib
import logging
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from BART import BART
from PIL import Image as PILImage
from torch.utils.data import DataLoader
from SimpsonsDataset import get_data, SimpsonsDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger("Batlogger (Train)")


# Reproducibility:
# It started when an alien device did what it did
# And stuck itself upon his wrist with secrets that it hid
# Now he's got superpowers, he's no ordinary kid He's Ben 10!
# Ben 10! => random_state=10
SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # for multi-GPU setups. I have no idea on what kinda setup this code will be run on...
torch.backends.cudnn.deterministic = True  # ensures deterministic convolution algorithms
torch.backends.cudnn.benchmark = False  # disables auto-tuning (which can introduce randomness)


# Ensures reproducibility in DataLoader workers
# I had not though about this before and will keep in mind for the future
def seed_worker(worker_id: int) -> None:
    # PEP 8 actually thinks we shall write 2**32 emphasizing the higher precedence, but I think that's just ugly
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser(description='Train BART model')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    args = parser.parse_args()

    BS = args.bs
    LR = args.lr
    EPOCHS = args.epochs

    images, labels, class_count = get_data('characters_train')
    logger.info(f"The number of images: {len(images)} is equal to the number of labels: {len(images) == len(labels)}")
    logger.info(f"Number of classes: {class_count}")

    img_shape_set: set[tuple[int, int, int]] = set()
    for img in images:
        w, h = img.size
        img_shape_set.add((h, w, 3))  # they do have RGB, but how PIL works is counterintuitive...

    if len(img_shape_set) == 1:
        logger.info(f"Dimensions of each image: {list(img_shape_set)[0]}")
    else:
        logger.info(f"There are {len(img_shape_set)} different sizes for images")
        logger.info(f"Different shapes found: {sorted(img_shape_set)[::len(img_shape_set) - 1]}")  # Get only the smallest and the largest image shapes


    # Based on the logging:
    #
    # There are 275 different sizes, all RGB, most images have 256 pixels in one dimension.
    # The images starting from (256, 256, 3) all the way to (1072, 1912, 3)
    #
    # Resizing is a MUST:
    # 275 different image sizes make batching impossible without resizing.
    # Unless you want to write some dumb out-of-this-world მატრაკვეცა case splits
    # Variable sizes would require padding/cropping, losing information unpredictably
    #
    # Why 128 x 128:
    # Computational efficiency: 128 x 128 = 16384 pixels vs 256 x 256 = 65536 pixels (4x less memory/computation)
    # Sufficient for character recognition: based on characters_illustration.png,
    # Simpsons characters have distinctive shapes that SHOULD survive downsampling
    # With only 16764 samples, smaller inputs reduce overfitting risk

    # resize all images to 128 x 128; keeps RGB channels
    images = np.array([np.array(img.resize((128, 128))) for img in images])

    # LabelEncoder converts categorical string labels into integer indices.
    # NNs require numerical inputs.
    # CrossEntropyLoss expects integer class indices as targets.
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=SEED)

    os.makedirs('inference_images', exist_ok=True)
    for i, img_array in enumerate(X_test):
        img = PILImage.fromarray(img_array)
        img.save(f'inference_images/pic_{i}.jpg')

    joblib.dump(label_encoder, 'label_encoder.joblib')
    logger.info("Saved label_encoder.joblib")
    logger.info(f"Saved {len(X_test)} test images to inference_images/")

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    # create datasets and dataloaders
    train_dataset = SimpsonsDataset(X_train, y_train)
    test_dataset = SimpsonsDataset(X_test, y_test)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f"Using CUDA")

    model = BART(num_classes=class_count).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    logger.info("Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            values, predicted = torch.max(outputs.data, dim=1)
            # initially, I had dim=0 but there were errors until I recalled that
            # dim=X means "reduce dimension X", not "operate on dimension X"...
            total += label.size(0)  # cause 16764 / BS = 16764 / 32 is not an integer, hence the last batch will not be of size 32.
            correct += (predicted == label).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device)
                label = label.to(device)

                outputs = model(img)
                values, predicted = torch.max(outputs.data, dim=1)
                val_total += label.size(0)
                val_correct += (predicted == label).sum().item()

        val_acc = 100 * val_correct / val_total

        logger.info(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.3f}, Train Acc: {train_acc:.3f}%, Val Acc: {val_acc:.3f}%")

    model.save()
    logger.info("BART-10M.pth SAVED AT CURRENT WORKING DIRECTORY")