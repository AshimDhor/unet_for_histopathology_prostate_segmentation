# UNet/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import random
from datetime import datetime

# Local imports
from model import UNet
from dataset import ProstateGlandDataset
from utils import (
    load_checkpoint,
    save_checkpoint,
    calculate_metrics,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 # Reduced batch size to prevent memory issues
NUM_EPOCHS = 25
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
IMAGE_SIZE = (512, 512) # Define image size here

# Dataset paths
TRAIN_IMG_DIR = "/home/ashim/Data_MS/2_Ring/train/IMAGES"
TRAIN_MASK_DIR = "/home/ashim/Data_MS/2_Ring/train/MANUAL GLANDS"
# We no longer use VAL_IMG_DIR for validation

def check_performance(loader, model, device="cuda"):
    print("Checking performance...")
    model.eval()
    all_metrics = {"dice": [], "iou": [], "accuracy": [], "precision": [], "recall": []}
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            preds = model(x)
            metrics = calculate_metrics(preds, y)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    model.train()
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    print(f"Validation Metrics:\n"
          f"  Dice Score: {avg_metrics['dice']:.4f}, IoU Score: {avg_metrics['iou']:.4f}\n"
          f"  Accuracy: {avg_metrics['accuracy']:.4f}, Precision: {avg_metrics['precision']:.4f}, Recall: {avg_metrics['recall']:.4f}")
    return avg_metrics['dice']

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Training")
    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device=DEVICE), targets.float().to(device=DEVICE)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

def main():
    # --- 1. SETUP FOR SYSTEMATIC SAVING ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("runs", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results and checkpoints will be saved in: {results_dir}")

    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 2. CREATE TRAIN/VALIDATION SPLIT ---
    all_train_images = sorted(os.listdir(TRAIN_IMG_DIR))
    random.seed(42) # for reproducibility
    random.shuffle(all_train_images)
    
    split_idx = int(len(all_train_images) * 0.8)
    train_filenames = all_train_images[:split_idx]
    val_filenames = all_train_images[split_idx:]
    print(f"Total training images: {len(train_filenames)}")
    print(f"Total validation images: {len(val_filenames)}")

    train_dataset = ProstateGlandDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        image_filenames=train_filenames,
        image_size=IMAGE_SIZE,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)

    val_dataset = ProstateGlandDataset(
        image_dir=TRAIN_IMG_DIR, # Still use train dir for images
        mask_dir=TRAIN_MASK_DIR, # Still use train dir for masks
        image_filenames=val_filenames,
        image_size=IMAGE_SIZE,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

    if LOAD_MODEL:
        load_checkpoint(torch.load("path/to/your/checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # Save model in the new results directory
        checkpoint_path = os.path.join(results_dir, "checkpoint.pth.tar")
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=checkpoint_path)
        
        check_performance(val_loader, model, device=DEVICE)

if __name__ == "__main__":
    main()