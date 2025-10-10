# UNet/test.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import torchvision

# Local imports
from model import UNet
from dataset import ProstateGlandDataset
from utils import load_checkpoint, calculate_metrics

# --- IMPORTANT: SET THESE PATHS ---
# Path to the trained model you want to test
MODEL_CHECKPOINT_PATH = "/home/ashim/ashim_MS/UNet/runs/2025-10-10_14-39-57/checkpoint.pth.tar"

# Directory where you want to save a few prediction examples
OUTPUT_DIR = "test_predictions"

# --- Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1 # Use a smaller batch size for testing if needed
NUM_WORKERS = 2
PIN_MEMORY = True
IMAGE_SIZE = (512, 512)

# --- Test Dataset Paths ---
TEST_IMG_DIR = "/home/ashim/Data_MS/2_Ring/test/IMAGES"
TEST_MASK_DIR = "/home/ashim/Data_MS/2_Ring/test/MANUAL GLANDS"


def test():
    print(f"Loading model from: {MODEL_CHECKPOINT_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Predicted images will be saved in: {OUTPUT_DIR}")

    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    load_checkpoint(torch.load(MODEL_CHECKPOINT_PATH), model)
    model.eval()

    # Create the test dataset
    test_filenames = sorted(os.listdir(TEST_IMG_DIR))
    test_dataset = ProstateGlandDataset(
        image_dir=TEST_IMG_DIR,
        mask_dir=TEST_MASK_DIR,
        image_filenames=test_filenames,
        image_size=IMAGE_SIZE,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    all_metrics = {"dice": [], "iou": [], "accuracy": [], "precision": [], "recall": []}
    
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader, desc="Testing")):
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            
            # Calculate metrics for the batch
            metrics = calculate_metrics(preds, y)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])

            # Save a few examples from the first batch
            if i == 0:
                preds_sigmoid = torch.sigmoid(preds)
                preds_binary = (preds_sigmoid > 0.5).float()
                # Save predictions and ground truth side-by-side
                comparison = torch.cat([y, preds_binary], dim=3)
                torchvision.utils.save_image(comparison, f"{OUTPUT_DIR}/comparison_batch_0.png")

    # Calculate and print average metrics over the entire test set
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    print("\n--- Final Test Set Performance ---")
    print(f"  Dice Score: {avg_metrics['dice']:.4f}")
    print(f"  IoU Score: {avg_metrics['iou']:.4f}")
    print(f"  Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall: {avg_metrics['recall']:.4f}")
    print("------------------------------------")


if __name__ == "__main__":
    test()