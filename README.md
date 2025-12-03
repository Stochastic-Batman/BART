# BART

CNN for Simpsons character classification

A Convolutional Neural Network built from scratch to classify images of Simpsons characters. 
The goal is to train a model that can accurately recognize multiple characters from the show, using a dataset organized by character and evaluated with the Macro F1 Score. No pre-trained models or transfer learning are allowed; 
the entire pipeline is custom-built.

The repository can be run as 2 Jupyter notebooks or as Python code, ultimately yielding the same output.

## Why "BART"?

The repo name "BART" is a reference to Bart Simpson, one of the main characters from The Simpsons. 
It also nods to the popular NLP architecture "BERT," blending the themes of deep learning and the Simpsons - even though BERT is not a vision model.

## Setup
Check your Python version with `python --version`. If it is not already Python 3.14, set it to 3.14. Then create a virtual environment with:

`python -m venv BART_venv`

and install requirements with:

`pip install -r requirements.txt`

## Project Structure (Jupyter Notebook)

## Project Structure (Python)

### `train.py`
Main training script. Handles:
- Data loading and preprocessing (resizing images to 128 x 128)
- Train/test split (80/20, `random_state=10` in honor of Ben 10)
- Training loop with validation
- Model saving

`train.py` uses:

> #### `BART.py`
> Defines the `BART` CNN architecture (and explains why this particular architecture was chosen):
> - 4 convolutional blocks (3 -> 32 -> 64 -> 128 -> 256 channels)
> - Each block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout
> - Classification head: Flatten -> FC(512) -> FC(256) -> FC(num_classes = 42 in our case)
> - ~9.7M parameters
>
> #### `SimpsonsDataset.py`
> - `get_data(directory)`: Loads PIL images and labels from directory structure
> - `SimpsonsDataset`: PyTorch Dataset class that converts images to tensors and normalizes them


## Running the code (Jupyter Notebook)

## Running the code (Python)

For training, run:

`python train.py --bs=32 --lr=0.001 --epochs=20`
