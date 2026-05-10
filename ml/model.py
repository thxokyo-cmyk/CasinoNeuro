"""
LSTM Neural Network for roulette number sequence prediction.
The model learns patterns in the sequence of numbers.
"""

import torch
import torch.nn as nn
import numpy as np


class RouletteLSTM(nn.Module):
    """
    LSTM-based model that takes a sequence of past numbers
    and predicts probability distribution over next number.
    """

    def __init__(self, input_size: int = 38, hidden_size: int = 128,
                 num_layers: int = 2, output_size: int = 38):
        """
        Args:
            input_size: 38 possible outcomes (0, 00, 1-36)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_size: 38 probability outputs
        """
        super(RouletteLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer: converts number index to dense vector
        self.embedding = nn.Embedding(input_size, 64)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size)
        )

        # Sector prediction head
        self.sector_fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 12)  # 12 sectors
        )

    def forward(self, x, hidden=None):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            hidden: Optional hidden state
        Returns:
            number_probs: Probability for each number (batch, 38)
            sector_probs: Probability for each sector (batch, 12)
        """
        # Embed input numbers
        embedded = self.embedding(x)  # (batch, seq_len, 64)

        # LSTM forward
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch, seq_len, hidden)

        # Attention
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)

        # Number prediction
        number_logits = self.fc(context)  # (batch, 38)

        # Sector prediction
        sector_logits = self.sector_fc(context)  # (batch, 12)

        return number_logits, sector_logits, hidden

    def predict_top_k(self, x, k=3):
        """Get top-k predictions with probabilities"""
        self.eval()
        with torch.no_grad():
            number_logits, sector_logits, _ = self.forward(x)

            # Softmax for probabilities
            number_probs = torch.softmax(number_logits, dim=-1)
            sector_probs = torch.softmax(sector_logits, dim=-1)

            # Top-k numbers
            top_number_probs, top_number_indices = torch.topk(number_probs, k, dim=-1)

            # Top-k sectors
            top_sector_probs, top_sector_indices = torch.topk(sector_probs, k, dim=-1)

        return (top_number_indices, top_number_probs,
                top_sector_indices, top_sector_probs)