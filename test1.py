import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from PIL import Image

# Local imports
from model import UNet
from dataset import ProstateGlandDataset
from utils import calculate_metrics

# --- CONFIGURATION ---
# The path to the specific training run you want to test
RUN_FOLDER = "/home/ashim/ashim_MS/UNet/runs/2025-10-10_14-39-57"

# Path to the trained model checkpoint
MODEL_PATH = os.path.join(RUN_FOLDER, "checkpoint.pth.tar")

# Path to your test dataset
TEST_IMG_DIR = "/home/ashim/Data_MS/2_Ring/test/IMAGES"
TEST_MASK_DIR = "/home/ashim/Data_MS/2_Ring/test/MANUAL GLANDS"

# Directory to save the predicted masks
PREDICTIONS_DIR = os.path.join(RUN_FOLDER, "test_predictions")

# Hyperparameters (should match training)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 # Can be the same or different from training
IMAGE_SIZE = (512, 512)
# --- END CONFIGURATION ---


def test_model():
    print(f"Loading model from: {MODEL_PATH}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    print(f"Predicted masks will be saved in: {PREDICTIONS_DIR}")

    # --- 1. Load Model ---
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval() # Set model to evaluation mode

    # --- 2. Create Test DataLoader ---
    test_image_filenames = sorted(os.listdir(TEST_IMG_DIR))
    
    test_dataset = ProstateGlandDataset(
        image_dir=TEST_IMG_DIR,
        mask_dir=TEST_MASK_DIR,
        image_filenames=test_image_filenames,
        image_size=IMAGE_SIZE,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # IMPORTANT: Do not shuffle test data
        num_workers=2,
        pin_memory=True,
    )

    # --- 3. Run Inference and Calculate Metrics ---
    all_metrics = {"dice": [], "iou": [], "accuracy": [], "precision": [], "recall": []}
    
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader, desc="Testing")):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Get predictions
            preds_logits = model(x)
            
            # Calculate metrics
            metrics = calculate_metrics(preds_logits, y)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])

            # --- 4. Save Predicted Masks ---
            preds_binary = (torch.sigmoid(preds_logits) > 0.5).float()
            
            # Move predictions to CPU and save each image in the batch
            preds_binary = preds_binary.cpu().numpy()
            
            start_idx = i * BATCH_SIZE
            for j in range(preds_binary.shape[0]):
                current_filename = test_image_filenames[start_idx + j]
                # Squeeze to remove channel and batch dims, scale to 0-255
                mask_to_save = (preds_binary[j].squeeze() * 255).astype(np.uint8)
                img = Image.fromarray(mask_to_save)
                img.save(os.path.join(PREDICTIONS_DIR, current_filename))

    # --- 5. Print Final Performance ---
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    print("\n--- Test Set Performance ---")
    print(f"  Dice Score: {avg_metrics['dice']:.4f}")
    print(f"  IoU Score: {avg_metrics['iou']:.4f}")
    print(f"  Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall: {avg_metrics['recall']:.4f}")
    print("----------------------------")


if __name__ == "__main__":
    test_model()