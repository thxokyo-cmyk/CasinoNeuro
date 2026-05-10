"""
Prediction engine - suggests 3 specific numbers + dozen sector.
"""

import torch
import numpy as np
from typing import List, Dict
from collections import Counter
from .model import RouletteLSTM
from .trainer import RouletteTrainer
from utils.roulette_logic import (
    number_to_index, index_to_number, get_color
)


# Three dozen sectors
DOZEN_SECTORS = {
    "1st12 (1-12)": list(range(1, 13)),
    "2nd12 (13-24)": list(range(13, 25)),
    "3rd12 (25-36)": list(range(25, 37)),
}


class RoulettePredictor:
    def __init__(self, trainer: RouletteTrainer):
        self.trainer = trainer
        self.model = trainer.model
        self.device = trainer.device
        self.sequence_length = trainer.sequence_length

    def predict(self, recent_numbers: List[str]) -> dict:
        """
        Given recent spin history, predict next outcomes.
        Returns top 3 numbers + best dozen sector.
        """
        if len(recent_numbers) < self.sequence_length:
            return self._fallback_prediction(recent_numbers)

        # Prepare input sequence
        sequence = recent_numbers[-self.sequence_length:]
        indices = [number_to_index(n) for n in sequence]
        x = torch.LongTensor([indices]).to(self.device)

        # Get predictions from model
        self.model.eval()
        with torch.no_grad():
            num_logits, sec_logits, _ = self.model(x)
            num_probs = torch.softmax(num_logits, dim=-1)[0]

        # Top numbers
        top_num_probs, top_num_idx = torch.topk(num_probs, 10)
        top_numbers = []
        for i in range(10):
            num_str = index_to_number(top_num_idx[i].item())
            prob = top_num_probs[i].item()
            top_numbers.append((num_str, prob))

        # Calculate dozen sector probabilities from number probs
        dozen_probs = self._calc_dozen_probs(num_probs)

        # Overall confidence
        uniform_prob = 1.0 / 38
        confidence = max(0, min((top_numbers[0][1] - uniform_prob) / uniform_prob, 1.0))

        return {
            "top_numbers": top_numbers[:3],
            "dozen_sector": dozen_probs,
            "confidence": confidence
        }

    def _calc_dozen_probs(self, num_probs: torch.Tensor) -> List[tuple]:
        """Calculate probability for each dozen from number probabilities"""
        dozen_results = []

        for name, numbers in DOZEN_SECTORS.items():
            # Sum probabilities of all numbers in this dozen
            prob_sum = 0.0
            for n in numbers:
                idx = number_to_index(str(n))
                prob_sum += num_probs[idx].item()
            dozen_results.append((name, prob_sum))

        # Sort by probability
        dozen_results.sort(key=lambda x: x[1], reverse=True)
        return dozen_results

    def _fallback_prediction(self, recent_numbers: List[str]) -> dict:
        """Fallback when not enough data - frequency and coverage analysis"""
        if not recent_numbers:
            return {
                "top_numbers": [("17", 0.027), ("23", 0.027), ("8", 0.027)],
                "dozen_sector": [
                    ("1st12 (1-12)", 0.333),
                    ("2nd12 (13-24)", 0.333),
                    ("3rd12 (25-36)", 0.333)
                ],
                "confidence": 0.0
            }

        # Frequency analysis
        freq = Counter(recent_numbers)
        total = len(recent_numbers)

        # Find hot numbers (appeared more than average)
        avg_freq = total / 38
        hot_numbers = [(n, c / total) for n, c in freq.most_common(10)
                       if n not in ("0", "00")]

        # Find cold numbers (not appeared or rarely appeared)
        all_nums = [str(i) for i in range(1, 37)]
        cold_numbers = [(n, 0.027) for n in all_nums if n not in freq]

        # Mix strategy: 2 hot + 1 cold (or vice versa depending on data)
        candidates = []
        if len(hot_numbers) >= 2:
            candidates.append(hot_numbers[0])
            candidates.append(hot_numbers[1])
        if cold_numbers:
            candidates.append(cold_numbers[0])
        elif len(hot_numbers) >= 3:
            candidates.append(hot_numbers[2])

        # Pad if needed
        while len(candidates) < 3:
            import random
            rand_num = str(random.randint(1, 36))
            candidates.append((rand_num, 0.027))

        top_numbers = candidates[:3]

        # Dozen analysis
        dozen_counts = {"1st12 (1-12)": 0, "2nd12 (13-24)": 0, "3rd12 (25-36)": 0}
        for n in recent_numbers:
            try:
                num = int(n)
                if 1 <= num <= 12:
                    dozen_counts["1st12 (1-12)"] += 1
                elif 13 <= num <= 24:
                    dozen_counts["2nd12 (13-24)"] += 1
                elif 25 <= num <= 36:
                    dozen_counts["3rd12 (25-36)"] += 1
            except ValueError:
                pass

        # Predict the dozen that hasn't appeared much (due for correction)
        dozen_total = sum(dozen_counts.values())
        if dozen_total > 0:
            dozen_probs = [(name, 1.0 - count / dozen_total)
                          for name, count in dozen_counts.items()]
        else:
            dozen_probs = [(name, 0.333) for name in dozen_counts]

        dozen_probs.sort(key=lambda x: x[1], reverse=True)

        return {
            "top_numbers": top_numbers,
            "dozen_sector": dozen_probs,
            "confidence": min(0.3, total / 50)
        }

    def get_recommendations(self, recent_numbers: List[str]) -> List[dict]:
        """
        Get final recommendations:
        - 3 specific numbers
        - 1 best dozen sector
        """
        prediction = self.predict(recent_numbers)

        recommendations = []

        # 3 number predictions
        for i, (num, prob) in enumerate(prediction["top_numbers"][:3]):
            num_int = int(num) if num != "00" else 0
            recommendations.append({
                "type": "number",
                "value": num,
                "probability": prob,
                "color": get_color(num_int),
                "rank": i + 1
            })

        # Best dozen sector
        if prediction["dozen_sector"]:
            best_dozen = prediction["dozen_sector"][0]
            recommendations.append({
                "type": "sector",
                "value": best_dozen[0],
                "probability": best_dozen[1],
                "color": "gold",
                "rank": 4
            })

        return recommendations, prediction["confidence"]