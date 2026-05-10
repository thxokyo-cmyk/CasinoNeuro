"""
ML Trainer with PyTorch GPU support.
Gracefully handles PyTorch unavailability while keeping full GPU support.
"""

import os
import sys
import numpy as np
from typing import List

from logger import log

# === FIX DLL LOADING BEFORE TORCH IMPORT ===
try:
    import fix_torch
except ImportError:
    pass

# === Import torch safely ===
TORCH_AVAILABLE = False
TORCH_ERROR = ""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        log.info("[ML] PyTorch GPU ready: " + gpu_name)
    else:
        log.info("[ML] PyTorch loaded (CPU mode)")

except Exception as e:
    TORCH_ERROR = str(e)
    log.error("[ML] PyTorch failed to load: " + TORCH_ERROR)


def number_to_index(number_str: str) -> int:
    if number_str == "00":
        return 37
    return int(number_str)


def index_to_number(index: int) -> str:
    if index == 37:
        return "00"
    return str(index)


# === LSTM Model (only defined if torch is available) ===

if TORCH_AVAILABLE:
    class RouletteLSTM(nn.Module):
        """LSTM model for roulette prediction"""

        def __init__(self, input_size=38, hidden_size=128, num_layers=2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.embedding = nn.Embedding(input_size, 64)
            self.lstm = nn.LSTM(64, hidden_size, num_layers,
                               batch_first=True, dropout=0.2)
            self.fc_number = nn.Linear(hidden_size, 38)
            self.fc_sector = nn.Linear(hidden_size, 3)

        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            last_output = lstm_out[:, -1, :]
            num_out = self.fc_number(last_output)
            sec_out = self.fc_sector(last_output)
            return num_out, sec_out, last_output
else:
    RouletteLSTM = None


class RouletteTrainer:
    """Training manager - PyTorch GPU when available, statistical fallback otherwise"""

    def __init__(self, config: dict):
        self.config = config
        self.sequence_length = config.get("sequence_length", 20)
        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 2)
        self.lr = config.get("learning_rate", 0.001)
        self.model_path = "roulette_model.pth"
        self.model = None
        self.device = None
        self.optimizer = None
        self.criterion = None

        if not TORCH_AVAILABLE:
            log.warning("[ML] ====================================")
            log.warning("[ML] PyTorch not available!")
            log.warning("[ML] Using STATISTICAL predictions (still works)")
            log.warning("[ML]")
            log.warning("[ML] TO ENABLE GPU AI:")
            log.warning("[ML] 1. Install Python 3.11 from python.org (NOT Microsoft Store)")
            log.warning("[ML] 2. pip install torch --index-url https://download.pytorch.org/whl/cu121")
            log.warning("[ML] ====================================")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("[ML] Device: " + str(self.device))

        self.model = RouletteLSTM(38, self.hidden_size, self.num_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self._load_model()

    def _load_model(self):
        if not TORCH_AVAILABLE or self.model is None:
            return
        if os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location=self.device,
                                  weights_only=True)
                self.model.load_state_dict(state)
                log.info("[ML] Loaded model from " + self.model_path)
            except Exception as e:
                log.warning("[ML] Could not load model: " + str(e))

    def train(self, numbers: List[str], epochs: int = 50) -> float:
        if not TORCH_AVAILABLE or self.model is None:
            log.info("[ML] No PyTorch - skipping training")
            return -1.0

        if len(numbers) < self.sequence_length + 1:
            return -1.0

        indices = [number_to_index(n) for n in numbers]
        X, y = [], []
        for i in range(len(indices) - self.sequence_length):
            X.append(indices[i:i + self.sequence_length])
            y.append(indices[i + self.sequence_length])

        X_tensor = torch.LongTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        last_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = 0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                num_out, sec_out, _ = self.model(batch_x)
                loss = self.criterion(num_out, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            last_loss = epoch_loss / max(batches, 1)
            if (epoch + 1) % 10 == 0:
                log.info("[ML] Epoch {}/{}, Loss: {:.4f}".format(epoch + 1, epochs, last_loss))

        self.save_model()
        return last_loss

    def predict_raw(self, numbers: List[str]):
        if not TORCH_AVAILABLE or self.model is None:
            return None

        if len(numbers) < self.sequence_length:
            return None

        sequence = numbers[-self.sequence_length:]
        indices = [number_to_index(n) for n in sequence]
        x = torch.LongTensor([indices]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            num_logits, sec_logits, _ = self.model(x)
            num_probs = torch.softmax(num_logits, dim=-1)[0]
            sec_probs = torch.softmax(sec_logits, dim=-1)[0]

        return {
            "num_probs": num_probs.cpu().numpy(),
            "sec_probs": sec_probs.cpu().numpy()
        }

    def save_model(self):
        if TORCH_AVAILABLE and self.model is not None:
            try:
                torch.save(self.model.state_dict(), self.model_path)
                log.info("[ML] Model saved")
            except Exception as e:
                log.error("[ML] Save error: " + str(e))

    def is_available(self) -> bool:
        """Check if ML training/prediction is available"""
        return TORCH_AVAILABLE and self.model is not None