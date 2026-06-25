from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch import nn

from .data import Example, Vocabulary


LABEL_ORDER = ["supports", "refutes", "nei"]


@dataclass
class Batch:
    claim_ids: torch.Tensor
    text_ids: torch.Tensor
    table_ids: torch.Tensor
    numeric_features: torch.Tensor
    labels: torch.Tensor | None
    example_ids: List[str]


class TensorizedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: Sequence[Example],
        vocabulary: Vocabulary,
        label_to_id: Dict[str, int],
        max_claim_length: int,
        max_text_length: int,
        max_table_length: int,
    ) -> None:
        self.examples = list(examples)
        self.vocabulary = vocabulary
        self.label_to_id = label_to_id
        self.max_claim_length = max_claim_length
        self.max_text_length = max_text_length
        self.max_table_length = max_table_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        example = self.examples[index]
        label_id = self.label_to_id.get(example.label) if example.label is not None else None
        return {
            "claim_ids": self.vocabulary.encode(example.claim_tokens, self.max_claim_length),
            "text_ids": self.vocabulary.encode(example.text_tokens, self.max_text_length),
            "table_ids": self.vocabulary.encode(example.table_tokens, self.max_table_length),
            "numeric_features": example.numeric_features,
            "label_id": label_id,
            "example_id": example.example_id,
        }


def collate_batch(items: Sequence[Dict[str, object]]) -> Batch:
    claim_ids = torch.tensor([item["claim_ids"] for item in items], dtype=torch.long)
    text_ids = torch.tensor([item["text_ids"] for item in items], dtype=torch.long)
    table_ids = torch.tensor([item["table_ids"] for item in items], dtype=torch.long)
    numeric_features = torch.tensor([item["numeric_features"] for item in items], dtype=torch.float32)
    label_values = [item["label_id"] for item in items]
    labels = None
    if all(value is not None for value in label_values):
        labels = torch.tensor(label_values, dtype=torch.long)
    example_ids = [str(item["example_id"]) for item in items]
    return Batch(
        claim_ids=claim_ids,
        text_ids=text_ids,
        table_ids=table_ids,
        numeric_features=numeric_features,
        labels=labels,
        example_ids=example_ids,
    )


class ClaimCaptionTableVerifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_numeric_features: int,
        num_labels: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.claim_proj = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.text_proj = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.table_proj = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        fusion_dim = hidden_dim * 6 + num_numeric_features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_labels),
        )

    def forward(
        self,
        claim_ids: torch.Tensor,
        text_ids: torch.Tensor,
        table_ids: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        claim_vec = self.claim_proj(self.masked_mean(claim_ids))
        text_vec = self.text_proj(self.masked_mean(text_ids))
        table_vec = self.table_proj(self.masked_mean(table_ids))
        evidence = torch.stack([text_vec, table_vec], dim=1)

        claim_query = claim_vec.unsqueeze(-1)
        scores = torch.matmul(evidence, claim_query).squeeze(-1) / math.sqrt(text_vec.size(-1))
        attention = torch.softmax(scores, dim=1)
        attended = torch.sum(evidence * attention.unsqueeze(-1), dim=1)

        fusion = torch.cat(
            [
                claim_vec,
                text_vec,
                table_vec,
                attended,
                torch.abs(text_vec - table_vec),
                text_vec * table_vec,
                numeric_features,
            ],
            dim=-1,
        )
        return self.fusion(fusion)

    def masked_mean(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        mask = (token_ids != 0).float().unsqueeze(-1)
        summed = torch.sum(embedded * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1.0)
        return summed / counts

