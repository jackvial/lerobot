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
    model_name: str = "gemini-2.5-pro-preview-05-06"

    # The high-level task instruction that will be passed to Gemini.
    prompt: str = ""

    # Number of actions (timesteps) to request from Gemini per query.
    n_action_steps: int = 10

    # ------------------------------------------------------------------
    # Gemini-specific control parameters
    # ------------------------------------------------------------------
    # Dimensionality of the robot action space (e.g. number of joints).
    # IMPORTANT: Set this via CLI with `--control.policy.action_dim=<int>` if
    # the default (6) does not match your robot.
    action_dim: int = 6

    # Explicitly define input/output features so that the config is
    # self-sufficient and does not rely on dataset/env parsing.
    input_features: dict = field(default_factory=dict)
    output_features: dict = field(default_factory=dict)

    # Leave observations and actions unnormalised – we expect them already to be
    # in interpretable units (degrees).
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # ------------------------------------------------------------------
    # Post-initialisation
    # ------------------------------------------------------------------

    def __post_init__(self):  # noqa: D401
        # Populate ACTION feature in `output_features` so that `make_policy`
        # sees this config as self-sufficient.
        from lerobot.configs.types import PolicyFeature, FeatureType

        if not isinstance(self.output_features, dict):
            self.output_features = {}

        self.output_features["action"] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(self.action_dim,)
        )

        # Ensure `input_features` exists as a dict even if empty – required by
        # the `PreTrainedPolicy` base class.
        if not isinstance(self.input_features, dict):
            self.input_features = {}

        # Call parent post-init _after_ features have been defined so that it
        # can derive helper attributes like `self.action_feature`.
        super().__post_init__()

        # Sanity-check that action_feature is correctly wired.
        if self.action_feature is None or self.action_feature.shape[0] != self.action_dim:
            raise ValueError(
                "GeminiConfig initialisation failed – action_feature missing or has incorrect shape. "
                f"Expected shape[0] == {self.action_dim}, got {getattr(self.action_feature, 'shape', None)}."
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
        # Ensure that an ACTION feature is present.
        if self.action_feature is None:
            raise ValueError("GeminiConfig requires an ACTION feature to be defined.")
        if self.action_feature.shape[0] != self.action_dim:
            raise ValueError(
                "GeminiConfig action_feature dimensionality does not match action_dim. "
                f"action_feature.shape[0] = {self.action_feature.shape[0]}, action_dim = {self.action_dim}"
            ) 