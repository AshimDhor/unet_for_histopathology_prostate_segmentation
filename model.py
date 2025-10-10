import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """A block containing two sequential convolutional layers."""
    # Each DoubleConv block consists of: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # If mid_channels is not provided, it defaults to out_channels.
        if not mid_channels:
            mid_channels = out_channels
        
        # Define the sequence of layers.
        self.double_conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # Second convolution
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Pass the input tensor through the sequential block.
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block for the encoder path."""
    # This block consists of a MaxPool2d layer followed by a DoubleConv block.

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # First, downsample the feature map by a factor of 2.
            nn.MaxPool2d(2),
            # Then, apply the DoubleConv block.
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # Pass the input tensor through the downscaling block.
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block for the decoder path."""
    # This block upscales the feature map and then combines it with a skip connection
    # from the encoder path, followed by a DoubleConv block.

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Choose the upsampling method.
        if bilinear:
            # Bilinear upsampling is faster but doesn't have learnable parameters.
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsampling, the number of channels is still in_channels.
            # We then concatenate with the skip connection, doubling the channels.
            # The input to the DoubleConv will be in_channels, and the output out_channels.
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Transposed convolution is a learnable upsampling method.
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # The DoubleConv block will have in_channels from the concatenated tensors.
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is the feature map from the previous decoder layer (to be upsampled).
        # x2 is the skip connection feature map from the corresponding encoder layer.

        # Upsample x1 to match the spatial dimensions of x2.
        x1 = self.up(x1)
        
        # In case of odd dimensions, max-pooling can lead to a size mismatch.
        # We need to pad the upsampled tensor (x1) to ensure its dimensions
        # are identical to the skip connection tensor (x2) before concatenation.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Apply padding to x1.
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate the skip connection (x2) and the upsampled tensor (x1) along the channel dimension.
        x = torch.cat([x2, x1], dim=1)
        
        # Pass the concatenated tensor through the DoubleConv block.
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution layer."""
    # This is a 1x1 convolution that maps the feature channels from the last decoder
    # block to the number of output classes.

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Apply the 1x1 convolution.
        return self.conv(x)


class UNet(nn.Module):
    """The main UNet model architecture."""
    # It consists of an encoder (downsampling path) and a decoder (upsampling path)
    # with skip connections between them.

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # --- Encoder Path (Downsampling) ---
        # Initial DoubleConv block.
        self.inc = DoubleConv(n_channels, 64)
        # Four downscaling blocks.
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # The 'factor' is used to adjust channel counts for bilinear vs. transposed conv.
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # --- Decoder Path (Upsampling) ---
        # Four upscaling blocks. They take the output from the previous decoder block
        # and the skip connection from the corresponding encoder block.
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Final output layer.
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # --- Encoder Path ---
        # The output of each encoder block is saved to be used as a skip connection.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # This is the bottleneck layer.

        # --- Decoder Path ---
        # The decoder reconstructs the segmentation map, using skip connections
        # to preserve high-resolution spatial information.
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # --- Final Output ---
        # The final 1x1 convolution produces the raw output (logits).
        # For a segmentation task, these logits represent the score for each class at each pixel.
        logits = self.outc(x)
        return logits
