"""
core/selection.py — Smooth Weighted Round-Robin algorithm (SWRR, NGINX style).

Pure function: builds the weighted rotation sequence only once;
the circular walk (cursor) stays in the LoadBalancer.
"""

from __future__ import annotations


def build_swrr_sequence(weights: dict[str, float]) -> list[str]:
    """
    Builds the interleaved weighted rotation list.

    E.g.: {"mistral": 2.0, "openai": 1.0, "google": 1.0} → sequence where mistral
    appears twice as often, without micro-bursts on a single provider.
    """
    if not weights:
        return []

    total_weight = sum(weights.values())

    names: list[str] = []
    slot_weights: list[int] = []
    for name, weight in weights.items():
        names.append(name)
        slot_weights.append(max(1, round((weight / total_weight) * 100)))

    total_slots = sum(slot_weights)
    current = [0] * len(names)
    sequence: list[str] = []

    for _ in range(total_slots):
        for i, w in enumerate(slot_weights):
            current[i] += w
        best = max(range(len(names)), key=lambda i: current[i])
        current[best] -= total_slots
        sequence.append(names[best])

    return sequence
