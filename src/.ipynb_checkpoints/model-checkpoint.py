import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HybridModel, self).__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(64)

        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch, seq_len, 1)
        x = x.permute(0, 2, 1)  # (batch, 1, seq_len) → (batch, channels, seq_len)
        x = self.cnn_block(x)   # (batch, 64, new_seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)

        attn_out, _ = self.attention(x, x, x)
        x = self.norm(attn_out + x)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 128)
        x = torch.mean(lstm_out, dim=1)  # GlobalAveragePooling1D

        return self.classifier(x)
