import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    """
    Two sequential conv layers each followed by ReLU, then max-pooling.
    Returns skip connection and pooled output.
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        skip = x
        pooled = self.pool(x)
        return skip, pooled

class TransUNetClassifier(nn.Module):
    """
    TransUNet-based classifier: encoder blocks + transformer on patches + FC head.
    Args:
        input_shape: tuple (C, H, W)
        num_classes: number of output classes
        patch_size: patch embedding size
        embed_dim: transformer embedding dimension
        num_heads: number of transformer heads
        ff_dim: feed-forward dimension in transformer
        num_transformer_blocks: number of transformer layers
    """
    def __init__(self, input_shape=(1,256,256), num_classes=4,
                 patch_size=4, embed_dim=128,
                 num_heads=4, ff_dim=256, num_transformer_blocks=4):
        super(TransUNetClassifier, self).__init__()
        in_channels = input_shape[0]
        # Encoder path
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        # Patch embedding via conv
        self.patch_size = patch_size
        self.patch_conv = nn.Conv2d(128, embed_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                        num_layers=num_transformer_blocks)
        # Classification head
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Encoder
        s1, p1 = self.encoder1(x)    # (B,64,H/2,W/2)
        s2, p2 = self.encoder2(p1)   # (B,128,H/4,W/4)
        B, C, H, W = p2.shape
        # Pad to divisible by patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            p2 = F.pad(p2, (0, pad_w, 0, pad_h))
        # Patch embedding
        patches = self.patch_conv(p2)            # (B,embed_dim,H',W')
        B, E, H_p, W_p = patches.shape
        num_patches = H_p * W_p
        patches = patches.view(B, E, num_patches).permute(2, 0, 1)
        # Transformer encoding
        trans_out = self.transformer_encoder(patches)  # (num_patches,B,embed_dim)
        pooled = trans_out.mean(dim=0)                 # (B,embed_dim)
        # Classification
        x = F.relu(self.fc1(pooled))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

if __name__ == "__main__":
    # quick sanity check
    model = TransUNetClassifier()
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print("Output shape:", out.shape)  # should be (2, num_classes)
