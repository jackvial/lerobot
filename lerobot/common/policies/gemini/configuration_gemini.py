#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("gemini")
@dataclass
class GeminiConfig(PreTrainedConfig):
    """Configuration for `GeminiPolicy`.

    This policy queries a Google Gemini chat model to obtain a sequence of joint
    targets given a natural-language *prompt* supplied at evaluation time, e.g.::

        --policy.prompt="Put nuts in bowl"

    Only inference is supported – training hooks are provided as no-ops so that
    the rest of the LeRobot pipeline can instantiate the config without error.
    """

    # Name of the Gemini model to call via the Google Generative-AI SDK or
    # `langchain_google_genai` wrapper.
    model_name: str = "gemini-1.5-pro-latest"

    # The high-level task instruction that will be passed to Gemini.
    prompt: str = ""

    # Number of actions (timesteps) to request from Gemini per query.
    n_action_steps: int = 10

    # Leave observations and actions unnormalised – we expect them already to be
    # in interpretable units (degrees).
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # ------------------------------------------------------------------
    # Mandatory abstract hooks from `PreTrainedConfig`
    # ------------------------------------------------------------------

    @property
    def observation_delta_indices(self):  # noqa: D401
        return None

    @property
    def action_delta_indices(self):  # noqa: D401
        return None

    @property
    def reward_delta_indices(self):  # noqa: D401
        return None

    def get_optimizer_preset(self) -> AdamConfig:  # type: ignore[override]
        # The model is not trainable – return a dummy preset.
        return AdamConfig(lr=1e-4)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:  # type: ignore[override]
        return None

    def validate_features(self) -> None:  # noqa: D401
        # Ensure that an ACTION feature (output shape) is available.  It will be
        # injected by `make_policy` based on either the dataset metadata or the
        # environment features.
        if self.action_feature is None:
            raise ValueError("GeminiConfig requires the policy to have an ACTION feature defined.") 