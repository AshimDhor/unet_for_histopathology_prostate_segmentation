# UNet for Histopathology Prostate Segmentation

## Overview

Implementation of U-Net architecture for automated segmentation of prostate glands in histopathological images. This project provides accurate gland boundary detection from H&E stained prostate biopsy specimens.

## Dataset

The dataset consists of whole-slide images of prostate biopsy specimens from 150 male patients (median age 63.2 years, range 41–86 years).

### Dataset Statistics
- **Total Images**: 1,500 (1000 training, 500 test)
- **Image Size**: 1500×1500 pixels
- **Magnification**: 100x (0.934 μm/pixel)
- **Staining**: H&E (Hematoxylin and Eosin)
- **Total Gland Annotations**: 18,851
- **Scanner**: Hamamatsu NanoZoomer S210

### Dataset Structure
```
Data_MS/2_Ring/
├── train/
│   ├── IMAGES/
│   └── MANUAL GLANDS/
└── test/
    ├── IMAGES/
    └── MANUAL GLANDS/
```

**Dataset Source**: https://data.mendeley.com/datasets/h8bdwrtnr5/1

## Model Architecture

U-Net with encoder-decoder structure:
- **Encoder**: 4 downsampling blocks with max pooling
- **Bottleneck**: 1024 channels at deepest layer
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Single channel binary segmentation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

## Installation

1. Clone the repository
```bash
git clone https://github.com/AshimDhor/unet_for_histopathology_prostate_segmentation.git
cd unet_for_histopathology_prostate_segmentation
```

2. Create conda environment
```bash
conda create -n MS python=3.8
conda activate MS
```

3. Install dependencies
```bash
pip install torch torchvision numpy pandas opencv-python pillow matplotlib scikit-learn tqdm
```

## Project Structure

```
├── model.py         # U-Net architecture
├── dataset.py       # Dataset and DataLoader
├── utils.py         # Loss functions and metrics
├── train.py         # Training script
├── test.py          # Testing script
├── test1.py         # Alternative test script
├── visualize.py     # Visualization tools
├── runs/            # Training runs directory
└── __pycache__/     # Python cache
```

## Usage

### Training

```bash
python train.py \
    --train_img_dir "/home/ashim/Data_MS/2_Ring/train/IMAGES" \
    --train_mask_dir "/home/ashim/Data_MS/2_Ring/train/MANUAL GLANDS" \
    --test_img_dir "/home/ashim/Data_MS/2_Ring/test/IMAGES" \
    --test_mask_dir "/home/ashim/Data_MS/2_Ring/test/MANUAL GLANDS" \
    --epochs 100 \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --loss combined
```

#### Training Parameters
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--loss`: Loss function [bce, dice, iou, combined] (default: combined)
- `--val_split`: Validation split ratio (default: 0.2)
- `--early_stopping_patience`: Early stopping patience (default: 15)

### Testing

```bash
python test.py \
    --model_path "./outputs/run_*/checkpoints/best_model.pth" \
    --test_img_dir "/home/ashim/Data_MS/2_Ring/test/IMAGES" \
    --test_mask_dir "/home/ashim/Data_MS/2_Ring/test/MANUAL GLANDS" \
    --save_predictions
```

### Visualization

```bash
python visualize.py \
    --model_path "./outputs/run_*/checkpoints/best_model.pth" \
    --mode batch \
    --num_samples 10 \
    --save_dir ./visualizations
```

## Features

- Data augmentation (rotation, flipping, brightness/contrast adjustments)
- Multiple loss functions (BCE, Dice, IoU, Combined)
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics
- GPU acceleration support
- Batch processing

## Evaluation Metrics

The model is evaluated using:
- Dice Score
- IoU (Jaccard Index)
- Precision
- Recall
- F1 Score
- Pixel Accuracy
- Specificity

## Output Structure

### Training Outputs
```
outputs/run_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best_model.pth
│   ├── final_model.pth
│   └── checkpoint_epoch_N.pth
├── config.json
└── history.npy
```

### Test Results
```
test_results/
├── predictions/
├── test_metrics.json
├── detailed_results.csv
└── all_metrics.npz
```

## Loss Functions

Available loss functions:
- **BCE**: Binary Cross-Entropy
- **Dice**: Dice coefficient loss
- **IoU**: Intersection over Union loss
- **Combined**: Weighted combination of BCE and Dice

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python train.py --batch_size 2
```

### Slow Training
Increase number of workers:
```bash
python train.py --num_workers 8
```

## Citation

If you use this code, please cite:

```bibtex
@misc{prostate_gland_segmentation_2024,
  title={UNet for Histopathology Prostate Segmentation},
  author={Ashim Dhor},
  year={2024},
  url={https://github.com/AshimDhor/unet_for_histopathology_prostate_segmentation}
}
```

Original U-Net paper:
```bibtex
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI 2015},
  pages={234--241},
  year={2015},
  publisher={Springer}
}
```

## License

MIT License

## Author

Ashim Dhor
