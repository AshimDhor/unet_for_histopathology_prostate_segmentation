# UNet/visualize.py

import torch
from model import UNet
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# --- IMPORTANT: SET THESE PATHS ---
MODEL_CHECKPOINT_PATH = "/home/ashim/ashim_MS/UNet/runs/2025-10-10_14-39-57/checkpoint.pth.tar"

# Choose any image from your TEST set to visualize
IMAGE_PATH = "/home/ashim/Data_MS/2_Ring/test/IMAGES/P15_D2_2_7_1.png" # Example image
MASK_PATH = "/home/ashim/Data_MS/2_Ring/test/MANUAL GLANDS/P15_D2_2_7_1.png" # Corresponding mask

# --- Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (512, 512)

def visualize():
    print(f"Loading model from: {MODEL_CHECKPOINT_PATH}")
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH)["state_dict"])
    model.eval()

    # Load original image and mask
    original_image = Image.open(IMAGE_PATH).convert("RGB")
    true_mask = Image.open(MASK_PATH).convert("L")

    # Define transforms for the single image
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    
    # Prepare image for model
    image_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = torch.sigmoid(model(image_tensor))
        prediction = (prediction > 0.5).cpu().numpy().squeeze()

    # Display the results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(true_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(prediction, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize()