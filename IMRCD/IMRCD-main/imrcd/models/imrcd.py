from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseCognitiveDiagnosisModel


class KANProjector(nn.Module):
    """Gated KAN projection used by the KC interaction module."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_basis: int = 3,
    ) -> None:
        super().__init__()
        if num_basis != 3:
            raise ValueError("The gated KAN projector requires three bases")

        self.num_basis = num_basis
        self.base_linear = nn.Linear(input_dim, output_dim)
        self.frequency1 = nn.Linear(input_dim, 1)
        self.frequency2 = nn.Linear(input_dim, 1)
        self.basis_gate = nn.Linear(input_dim, num_basis)

        # This parameter is retained to reproduce the initialization stream of
        # the model used in the reported experiments. The gated formulation
        # below uses input-dependent gates directly.
        self.basis_weights = nn.Parameter(
            torch.empty(output_dim, num_basis)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.base_linear.weight)
        nn.init.zeros_(self.base_linear.bias)
        nn.init.xavier_uniform_(self.frequency1.weight)
        nn.init.xavier_uniform_(self.frequency2.weight)
        nn.init.xavier_uniform_(self.basis_gate.weight)
        nn.init.zeros_(self.frequency1.bias)
        nn.init.zeros_(self.frequency2.bias)
        nn.init.zeros_(self.basis_gate.bias)
        nn.init.normal_(self.basis_weights, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(inputs)
        frequency1 = self.frequency1(inputs)
        frequency2 = self.frequency2(inputs)
        gates = torch.sigmoid(self.basis_gate(inputs))

        sin_basis = torch.sin(base * math.pi * frequency1)
        cos_basis = torch.cos(base * math.pi * frequency2)
        silu_basis = F.silu(base)
        return (
            base
            + gates[..., 0:1] * sin_basis
            + gates[..., 1:2] * cos_basis
            + gates[..., 2:3] * silu_basis
        )


class InteractionMemoryBank(nn.Module):
    """Learnable interaction memory with grouped key-value slots."""

    def __init__(
        self,
        interaction_dim: int,
        query_dim: int,
        memory_dim: int = 32,
        memory_groups: int = 4,
        memory_slots: int = 32,
        residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.interaction_dim = interaction_dim
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.memory_groups = memory_groups
        self.memory_slots = memory_slots
        self.residual_scale = residual_scale

        self.query_projection = nn.Linear(
            query_dim, memory_groups * memory_dim
        )
        self.key_bank = nn.Parameter(
            torch.randn(memory_groups, memory_slots, memory_dim) * 0.02
        )
        self.value_bank = nn.Parameter(
            torch.randn(memory_groups, memory_slots, memory_dim) * 0.02
        )
        self.group_gate = nn.Linear(query_dim, memory_groups)
        self.output_mapping = nn.Sequential(
            nn.Linear(
                memory_groups * memory_dim + interaction_dim,
                memory_dim,
            ),
            nn.LayerNorm(memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, interaction_dim),
        )
        self.update_gate = nn.Linear(
            memory_groups * memory_dim + interaction_dim, 1
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.zeros_(self.query_projection.bias)
        nn.init.xavier_uniform_(self.group_gate.weight)
        nn.init.zeros_(self.group_gate.bias)
        nn.init.normal_(self.key_bank, std=0.02)
        nn.init.normal_(self.value_bank, std=0.02)

        for module in self.output_mapping:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.normal_(
            self.output_mapping[-1].weight, mean=0.0, std=1e-4
        )
        nn.init.zeros_(self.output_mapping[-1].bias)
        nn.init.xavier_uniform_(self.update_gate.weight)
        nn.init.zeros_(self.update_gate.bias)

    def forward(
        self,
        base_interaction: torch.Tensor,
        memory_query: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = base_interaction.size(0)
        queries = self.query_projection(memory_query).view(
            batch_size, self.memory_groups, self.memory_dim
        )
        matching_scores = torch.einsum(
            "bgd,gsd->bgs", queries, self.key_bank
        ) / math.sqrt(self.memory_dim)
        slot_weights = torch.softmax(matching_scores, dim=-1)
        group_readouts = torch.einsum(
            "bgs,gsd->bgd", slot_weights, self.value_bank
        )
        group_weights = torch.softmax(
            self.group_gate(memory_query), dim=-1
        ).unsqueeze(-1)
        memory_readout = (group_readouts * group_weights).reshape(
            batch_size, self.memory_groups * self.memory_dim
        )

        combined = torch.cat([base_interaction, memory_readout], dim=-1)
        memory_update = self.output_mapping(combined)
        update_weight = torch.sigmoid(self.update_gate(combined))
        return (
            base_interaction
            + self.residual_scale * update_weight * memory_update
        )


class MemoryGuidedKCInteraction(nn.Module):
    """Generate KC updates from the exercise and retrieved memory."""

    def __init__(
        self,
        embedding_dim: int,
        concept_count: int,
        interaction_dim: int = 11,
        hidden_dim: int = 64,
        interaction_heads: int = 4,
        num_basis: int = 3,
        memory_dim: int = 32,
        memory_groups: int = 4,
        memory_slots: int = 32,
        residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden_dim % interaction_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by interaction_heads"
            )

        self.hidden_dim = hidden_dim
        self.interaction_heads = interaction_heads
        self.head_dim = hidden_dim // interaction_heads
        self.concept_count = concept_count

        # Keep this registration order aligned with the model used in the
        # reported experiments because it determines seeded initialization.
        self.interaction_projection = KANProjector(
            embedding_dim + interaction_dim,
            hidden_dim,
            num_basis=num_basis,
        )
        self.kc_projection = KANProjector(
            embedding_dim,
            hidden_dim,
            num_basis=num_basis,
        )
        self.interaction_memory = InteractionMemoryBank(
            interaction_dim=interaction_dim,
            query_dim=interaction_dim,
            memory_dim=memory_dim,
            memory_groups=memory_groups,
            memory_slots=memory_slots,
            residual_scale=residual_scale,
        )
        self.interaction_dropout = nn.Dropout(0.01)
        self.q_projection = nn.Linear(1, hidden_dim)
        self.output_mapping = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.normal_(
            self.output_mapping[-1].weight, mean=0.0, std=1e-4
        )
        nn.init.zeros_(self.output_mapping[-1].bias)

    def forward(
        self,
        static_kc: torch.Tensor,
        exercise_embedding: torch.Tensor,
        base_interaction: torch.Tensor,
        q_vectors: torch.Tensor,
        memory_query: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = static_kc.size(0)
        concept_count = static_kc.size(1)

        retrieved_memory = self.interaction_memory(
            base_interaction, memory_query
        )
        interaction_input = torch.cat(
            [exercise_embedding.detach(), retrieved_memory], dim=-1
        )
        interaction_vector = self.interaction_projection(
            interaction_input
        ).unsqueeze(1)
        projected_kc = self.kc_projection(static_kc)

        interaction_heads = interaction_vector.view(
            batch_size, 1, self.interaction_heads, self.head_dim
        )
        kc_heads = projected_kc.view(
            batch_size,
            concept_count,
            self.interaction_heads,
            self.head_dim,
        )
        headwise_interaction = interaction_heads * kc_heads

        q_mask = q_vectors.view(batch_size, concept_count, 1, 1)
        q_features = self.q_projection(q_vectors.unsqueeze(-1))
        q_heads = q_features.view(
            batch_size,
            concept_count,
            self.interaction_heads,
            self.head_dim,
        )
        constrained_interaction = (
            headwise_interaction + q_heads
        ) * q_mask
        combined_heads = constrained_interaction.view(
            batch_size, concept_count, self.hidden_dim
        )
        return self.output_mapping(
            self.interaction_dropout(combined_heads)
        )


class IMRCD(BaseCognitiveDiagnosisModel):
    """Interaction Memory Retrieval for Cognitive Diagnosis."""

    def __init__(
        self,
        student_count: int,
        exercise_count: int,
        concept_count: int,
        embedding_dim: int = 100,
        interaction_dim: int = 11,
        interaction_hidden_dim: int = 64,
        interaction_heads: int = 4,
        num_basis: int = 3,
        memory_dim: int = 32,
        memory_groups: int = 4,
        memory_slots: int = 32,
        residual_scale: float = 0.05,
        query_gamma_max: float = 0.05,
    ) -> None:
        super().__init__()
        self.student_count = student_count
        self.exercise_count = exercise_count
        self.concept_count = concept_count
        self.embedding_dim = embedding_dim
        self.interaction_dim = interaction_dim
        self.num_basis = num_basis
        self.query_gamma_max = query_gamma_max

        # Base interaction is intentionally registered before the embeddings
        # to preserve the seeded initialization used by the original runs.
        self.base_interaction = nn.Parameter(
            torch.empty(interaction_dim)
        )
        query_condition_dim = min(2, embedding_dim)

        self.student_embedding = nn.Embedding(
            student_count, embedding_dim
        )
        self.exercise_embedding = nn.Embedding(
            exercise_count, embedding_dim
        )
        self.static_kc = nn.Parameter(
            torch.zeros(concept_count, embedding_dim)
        )
        self.student_query_projection = nn.Linear(
            embedding_dim, query_condition_dim, bias=False
        )
        self.exercise_query_projection = nn.Linear(
            embedding_dim, query_condition_dim, bias=False
        )
        self.query_fusion = nn.Linear(
            interaction_dim + 2 * query_condition_dim,
            interaction_dim,
        )
        self.query_gamma = nn.Parameter(torch.tensor(-4.0))

        self.embedding_dropout = nn.Dropout(0.2)
        nn.init.kaiming_normal_(self.static_kc)

        prediction_input_dim = 2 * embedding_dim
        self.student_prediction = nn.Linear(
            prediction_input_dim, concept_count
        )
        self.exercise_prediction = nn.Linear(
            prediction_input_dim, concept_count
        )
        self.score_mapping = nn.Linear(concept_count, 1)

        self.kc_interaction = MemoryGuidedKCInteraction(
            embedding_dim=embedding_dim,
            concept_count=concept_count,
            interaction_dim=interaction_dim,
            hidden_dim=interaction_hidden_dim,
            interaction_heads=interaction_heads,
            num_basis=num_basis,
            memory_dim=memory_dim,
            memory_groups=memory_groups,
            memory_slots=memory_slots,
            residual_scale=residual_scale,
        )
        self.fusion_strength = nn.Parameter(torch.tensor(0.1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dim() >= 2 and "weight" in name:
                nn.init.xavier_normal_(parameter)

        nn.init.xavier_normal_(self.static_kc)
        nn.init.normal_(self.base_interaction, mean=0.0, std=0.02)
        nn.init.zeros_(self.query_fusion.bias)

    def effective_query_gamma(self) -> torch.Tensor:
        return self.query_gamma_max * torch.sigmoid(self.query_gamma)

    def forward(
        self,
        student_ids: torch.LongTensor,
        exercise_ids: torch.LongTensor,
        q_vectors: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = student_ids.size(0)
        student_embedding = self.student_embedding(student_ids)
        exercise_embedding = self.exercise_embedding(exercise_ids)
        student_main = self.embedding_dropout(student_embedding)
        exercise_main = self.embedding_dropout(exercise_embedding)

        base_interaction = self.base_interaction.unsqueeze(0).expand(
            batch_size, -1
        )
        student_query = torch.tanh(
            self.student_query_projection(student_embedding.detach())
        )
        exercise_query = torch.tanh(
            self.exercise_query_projection(exercise_embedding.detach())
        )
        query_input = torch.cat(
            [student_query, exercise_query, base_interaction], dim=-1
        )
        query_update = torch.tanh(self.query_fusion(query_input))
        memory_query = (
            base_interaction
            + self.effective_query_gamma() * query_update
        )

        static_kc = self.static_kc.unsqueeze(0).repeat(
            batch_size, 1, 1
        )
        kc_update = self.kc_interaction(
            static_kc,
            exercise_main,
            base_interaction,
            q_vectors,
            memory_query,
        )
        enhanced_kc = (
            static_kc + F.softplus(self.fusion_strength) * kc_update
        )

        student_expanded = student_main.unsqueeze(1).repeat(
            1, self.concept_count, 1
        )
        exercise_expanded = exercise_main.unsqueeze(1).repeat(
            1, self.concept_count, 1
        )
        student_input = torch.cat(
            [student_expanded, enhanced_kc], dim=-1
        )
        exercise_input = torch.cat(
            [exercise_expanded, enhanced_kc], dim=-1
        )

        student_diagnosis = torch.sigmoid(
            self.student_prediction(student_input)
        )
        exercise_diagnosis = torch.sigmoid(
            self.exercise_prediction(exercise_input)
        )
        kc_scores = self.score_mapping(
            student_diagnosis - exercise_diagnosis
        ).squeeze(-1)

        weighted_scores = kc_scores * q_vectors
        denominator = q_vectors.sum(dim=1, keepdim=True) + 1e-8
        response_logits = (
            weighted_scores.sum(dim=1, keepdim=True) / denominator
        )
        return torch.sigmoid(response_logits).squeeze(-1)
