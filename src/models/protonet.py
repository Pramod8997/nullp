import torch
import torch.nn as nn
from typing import Dict, Optional

class ProtoNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, embedding_size: int = 64):
        super(ProtoNet, self).__init__()
        # Simple CNN encoder for 1D power sequences
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * (input_size // 4), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: [batch_size, 1, input_size]
        return self.encoder(x)

class SupportSetManager:
    def __init__(self, max_memory_per_class: int = 20):
        self.max_memory_per_class = max_memory_per_class
        # Mapping class_id -> tensor of embeddings [N, embedding_size]
        self.support_sets: Dict[str, torch.Tensor] = {}

    def add_embedding(self, class_id: str, embedding: torch.Tensor) -> None:
        """
        Adds an embedding to the support set. 
        Implements [INV-4]: indices [0, 1, 2] are anchors. Eviction applies to [3:19].
        """
        embedding = embedding.detach() # Ensure no grad
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
            
        if class_id not in self.support_sets:
            self.support_sets[class_id] = embedding
            return

        current_set = self.support_sets[class_id]
        if current_set.size(0) < self.max_memory_per_class:
            self.support_sets[class_id] = torch.cat([current_set, embedding], dim=0)
        else:
            # Evict the oldest non-anchor (index 3) and append the new one
            # Anchors are indices 0, 1, 2. We keep them, remove index 3, and append at the end.
            anchors = current_set[:3]
            tail = current_set[4:]
            self.support_sets[class_id] = torch.cat([anchors, tail, embedding], dim=0)

    def get_prototype(self, class_id: str) -> Optional[torch.Tensor]:
        if class_id not in self.support_sets:
            return None
        return self.support_sets[class_id].mean(dim=0)
