# Plant Leaf Disease Detection (CNN)

## Problem
Plant diseases reduce crop yield, and diagnosing them by eye needs an expert.
This project trains a CNN from scratch that looks at a photo of a leaf of **one crop**
and classifies it as healthy or as one of that crop's diseases.

## Dataset
- **PlantVillage** (Mohanty et al., 2016), loaded from Hugging Face:
  `BrandonFors/Plant-Diseases-PlantVillage-Dataset`
- RGB leaf images (256x256), resized to 128x128.
- One crop only (set by `CROP` in the notebook, default `Tomato`): its disease classes + healthy.
- Splits: the dataset's own train/test split; 15% of train is used for validation.

| Item | Value |
|---|---|
| Crop | Tomato |
| Number of classes | 10 (9 diseases + healthy) |
| Train images | 12,353 |
| Validation images | 2,179 |
| Test images | 3,628 |

## Architecture
Custom CNN, no pretrained weights:

| Part | Layers |
|---|---|
| Block 1 | Conv(3->32, 3x3) - ReLU - MaxPool(2) |
| Block 2 | Conv(32->64, 3x3) - ReLU - MaxPool(2) |
| Block 3 | Conv(64->128, 3x3) - ReLU - MaxPool(2) |
| Classifier | Flatten - Linear(32768->256) - ReLU - Dropout(0.3) - Linear(256->classes) |

Loss: CrossEntropy. Optimizer: Adam (lr = 0.001). Batch size 64, 15 epochs.
The trained model is saved to `model.pth`.

## Results
| Metric | Value |
|---|---|
| Validation accuracy (last epoch) | 94.54% |
| Test accuracy | 94.43% |

Training curves (train/val accuracy and loss):

![Training curves](training_curves.jpg)

## How to run
1. Open `CNN_PlantVillage.ipynb` in Google Colab and enable a GPU
   (Runtime > Change runtime type > GPU).
2. Run all cells. The notebook downloads the dataset from Hugging Face, trains the CNN
   for 15 epochs, plots the accuracy/loss curves, evaluates on the test set
   and saves the model to `model.pth`.

To run locally: `pip install -r requirements.txt`, then open the notebook in Jupyter.

To load the saved model (after running the cell that defines the `CNN` class):
```python
checkpoint = torch.load("model.pth", map_location="cpu")
model = CNN(len(checkpoint["class_names"]))
model.load_state_dict(checkpoint["model_state"])
model.eval()
```
