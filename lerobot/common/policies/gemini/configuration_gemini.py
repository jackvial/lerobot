#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Any

from lerobot.common.optim.optimizers import AdamConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("gemini")
@dataclass
class GeminiConfig(PreTrainedConfig):
    """Configuration for GeminiPolicy.

    This policy proxies action selection to the Google Gemini LLM.  It is
    stateless and therefore has no trainable parameters, but we still expose a
    few settings so that users can configure prompting.

    Args:
        prompt: High-level instruction describing the task (e.g. "Put nuts in bowl").
        model: Gemini model identifier to use via the Google Generative AI API.
        temperature: Sampling temperature passed to the model.
        n_action_steps: When the LLM returns a sequence of actions we can buffer
            more than one and play them out over successive environment steps.
    """

    prompt: str = "Put nuts in bowl"
    model: str = "gemini-1.5-pro"
    temperature: float = 0.0
    n_action_steps: int = 1

    # ---------------------------------------------------------------------
    # Normalisation: We leave observations untouched and expect ACTION to be in
    # [-1, 1] so that it is compatible with typical LeRobot environments.
    # ---------------------------------------------------------------------
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # ------------------------------------------------------------------
    # Mandatory abstract-method implementations from PreTrainedConfig
    # ------------------------------------------------------------------
    @property
    def observation_delta_indices(self) -> list | None:  # noqa: D401
        return None

    @property
    def action_delta_indices(self) -> list | None:  # noqa: D401
        return None

    @property
    def reward_delta_indices(self) -> list | None:  # noqa: D401
        return None

    # GeminiPolicy is not trainable, but the training pipeline expects an
    # optimiser config.  We therefore return a dummy preset.
    def get_optimizer_preset(self) -> AdamConfig:  # type: ignore[override]
        return AdamConfig(lr=1e-4)

    def get_scheduler_preset(self):  # type: ignore[override]
        return None

    def validate_features(self) -> None:  # type: ignore[override]
        if self.action_feature is None:
            raise ValueError("GeminiPolicy requires an ACTION feature to be defined.") 