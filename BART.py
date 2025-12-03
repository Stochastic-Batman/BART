import torch
import torch.nn as nn


# Why this architecture?
#
# Quick overview:
# Convolution blocks: Extract features -> spatial patterns
# FC layers: Learn complex decision boundaries between classes
# Final Linear layer: Produces raw scores (logits) for each of the 42 class
# Softmax: Applied externally through nn.CrossEntropyLoss()
#
# Depth of 4 convolution blocks
# Block 1: Learns simple edges, colors, basic textures
# Block 2: Combines edges into simple shapes -> curves, corners
# Block 3: Recognizes more complex patterns -> facial features, clothing patterns
# Block 4: Understands high-level features -> faces, body parts, character-specific details
# Besides, I think for 16764 samples, <3 layers will not be enough and >5 will overfit
#
# My summary from Fei-Fei Li's CNN slides
# As we go deeper through the network, two things happen simultaneously but in opposite directions:
# The spatial dimensions shrink due to pooling (in our case, 128 x 128 -> 64 x 64 -> 32 x 32 -> 16 x 16 -> 8 x 8)
# meaning we lose precise location information about where features appear in the image.
# At the same time, the channel dimensions grow (3 -> 32 -> 64 -> 128 -> 256),
# meaning we gain richer and more abstract semantic representations at each location.
# The first law of alchemy is the Law of Equivalent Exchange:
# early layers with high resolution and few channels detect simple patterns like edges at specific locations,
# while later layers with low resolution and many channels recognize complex concepts like facial features.
#
# Dropout: 0.25 -> 0.25 -> 0.4 -> 0.4 -> 0.5:
# Reasoning is that when we increase dimensions overfitting risk is higher
#
# Two FC layers  at the end (512 -> 256 -> num_classes):
# 1 FC layer might be too simple for decision boundary, given we have 42 classes
# >3 FC layers risk overfitting with this data amount
#
# Calculating the number of parameters (this includes the whole data, before train/test split)
# Given:
# C_in = Number of input channels
# K_h = Height of the kernel
# K_w = Width of the kernel
# C_out = Number of output channels
# N_in = Number of input features
# N_out = Number of output features
# + whenever you see "+ 1" outta nowhere (RKO!) it's for bias term.
# Then, the formulas for calculating the number of parameters is:
# For Conv2d -> (C_in x K_h x K_w + 1) x C_out
# For BatchNorm2d -> 2 x C_out (one for mean, one for scale)
# For Linear -> (N_in x N_out) + N_out
#
# Block 1:
# Conv2d(3 -> 32, 3 x 3) -> (3 x 3 x 3 + 1) x 32 = 896
# BatchNorm2d(32) -> 32 x 2 = 64
# Conv2d(32 -> 32, 3 x 3) -> (3 x 3 x 32 + 1) x 32 = 9248
# BatchNorm2d(32) -> 64
# Block 1 total: 10272
#
# Block 2:
# Conv2d(32 -> 64, 3 x 3) -> (3 x 3 x 32 + 1) x 64 = 18496
# BatchNorm2d(64) -> 128
# Conv2d(64 -> 64, 3 x 3) -> (3 x 3 x 64 + 1) x 64 = 36928
# BatchNorm2d(64) -> 128
# Block 2 total: 55680
#
# Block 3:
# Conv2d(64 -> 128, 3 x 3) -> (3 x 3 x 64 + 1) x 128 = 73856
# BatchNorm2d(128) -> 256
# Conv2d(128 -> 128, 3 x 3) -> (3 x 3 x 128 + 1) x 128 = 147584
# BatchNorm2d(128) -> 256
# Block 3 total: 221952
#
# Block 4:
# Conv2d(128 -> 256, 3 x 3) -> (3 x 3 x 128 + 1) x 256 = 295168
# BatchNorm2d(256) -> 512
# Conv2d(256 -> 256, 3 x 3) -> (3 x 3 x 256 + 1) x 256 = 590080
# BatchNorm2d(256) -> 512
# Block 4 total: 886272
#
# So 1174176 parameters for 4 convolutional blocks
#
# Classification Head (the heavyweight!):
# Linear(16384 -> 512) -> 16384 x 512 + 512 = 8389120
# Linear(512 -> 256) -> 512 x 256 + 256 = 131328
# Linear(256 -> 42 as we have 42 classes) -> 256 x 42 + 42 = 10794
# FC total: 8531242
#
# Total: 8531242 + 1174176 = 9705418 parameters, ratio of 16764 / 9705418 ~ 0.00173 samples per parameter, which is quite low.
# # I think ~10 samples per parameter would be sufficient, but we have no data (even if you were to add that remaining 20% of the dataset currently not in characters_train/)
# ~87% of parameters are in the first FC layer alone => most parameters are in the dense layers, not the convolutional ones.


class BART(nn.Module):
    def __init__(self, num_classes=42):
        super(BART, self).__init__()

        # Block 1: (128, 128, 3) -> (64, 64, 32)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'),  # 'same' maintains input size after convolution
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # save memory by modifying the input tensor directly,
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # at the risk of losing the original data
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        # Block 2: (64, 64, 32) -> (32, 32, 64)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        # Block 3: (32, 32, 64) -> (16, 16, 128)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.4)
        )

        # Block 4: (16, 16, 128) -> (8, 8, 256)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.4)
        )

        # Classification Head: (8*8*256) -> num_classes
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classification_head(x)
        return x


    def predict(self, x: torch.Tensor) -> torch.Tensor:  # predict class, just in case
        return torch.argmax(self.forward(x), dim=1)


    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:  # get class distribution, just in case
        return torch.softmax(self.forward(x), dim=1)


    def save(self, path: str = "") -> None:
        model_name = 'BART-10M.pth'
        torch.save(self.state_dict(), f"{path}/{model_name}" if path else model_name)  # "works on my machine" (Linux)