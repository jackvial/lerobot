#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import OrderedDict

from torch import nn

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.envs.configs import EnvConfig
from lerobot.common.envs.utils import env_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType

logger = logging.getLogger(__name__)

# Maps policy type to its class.
POLICY_MAPPING = OrderedDict()
# Maps policy type to its config class.
POLICY_CONFIG_MAPPING = OrderedDict()


def register_policy(name: str, config_cls: type, policy_cls: type):
    POLICY_MAPPING[name] = policy_cls
    POLICY_CONFIG_MAPPING[name] = config_cls


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "tdmpc":
        from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "act":
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "vqbet":
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy
    elif name == "pi0":
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi0fast":
        from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

        return PI0FASTPolicy
    elif name == "gemini":
        from lerobot.common.policies.gemini.modeling_gemini import GeminiPolicy

        return GeminiPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "tdmpc":
        return TDMPCConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "vqbet":
        return VQBeTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi0fast":
        return PI0FASTConfig(**kwargs)
    elif policy_type == "gemini":
        from lerobot.common.policies.gemini.configuration_gemini import GeminiConfig

        return GeminiConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
) -> nn.Module:
    """Makes a policy and its config from a path, and if provided, a dataset or an env config.

    If a path to a pretrained model is provided, the config is loaded from the repo or local path.
    Then, if a dataset metadata or an env config is provided, the config is updated from it.
    For instance, the observation and action spaces are inferred from the dataset or env config.
    Finally, the policy is instantiated from the config. If a path to a pretrained model is provided,
    the weights are loaded from the repo or local path.

    Order of precedence for config loading & updates:
    1. Values stored in the local/hub `config.json` (if `cfg.pretrained_path` is provided).
    2. Values provided by `ds_meta` or `env_cfg` (e.g. observation & action spaces).
    3. Values provided by `cfg` attributes (i.e. CLI overrides).

    Args:
        cfg: The policy config. Can be a PreTrainedConfig or a policy-specific config.
        ds_meta: The dataset metadata. Used to infer observation and action spaces.
        env_cfg: The env config. Used to infer observation and action spaces.

    Returns:
        A policy instance.
    """
    if ds_meta is not None and env_cfg is not None:
        raise ValueError("Only one of a dataset metadata or a sim env should be provided.")

    # Determine if the provided policy config `cfg` might already have its features defined.
    # This is usually the case if `cfg` was loaded from a pretrained model's config.json
    # or if the policy type inherently defines its features (like a future GeminiConfig might).
    # A basic check: output_features (action) exists and has a shape.
    # input_features (observation) is a dict; an empty dict can be valid if a policy needs no specific obs.
    has_defined_output_feature = False
    if hasattr(cfg, 'output_features') and isinstance(cfg.output_features, dict) and len(cfg.output_features) > 0:
        # Check if at least one ACTION feature has a non-empty shape.
        for ft in cfg.output_features.values():
            if ft.type is FeatureType.ACTION and getattr(ft, 'shape', None):
                has_defined_output_feature = True
                break
    has_defined_input_features = hasattr(cfg, 'input_features') and isinstance(cfg.input_features, dict)
    # For Gemini, action_feature is critical, observation_features might be less so for the base class.
    # Let's consider it self-sufficient if output_features looks okay.
    is_config_self_sufficient = has_defined_output_feature

    # Special-case fallback: GeminiConfig may arrive with empty features when built via CLI only. If so,
    # auto-populate a default ACTION feature from its `action_dim` to make it self-sufficient.
    if cfg.type == "gemini" and not is_config_self_sufficient:
        try:
            from lerobot.configs.types import PolicyFeature
            # Only add if not already present
            if not isinstance(cfg.output_features, dict):
                cfg.output_features = {}
            cfg.output_features["action"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(getattr(cfg, "action_dim", 6),),
            )
            is_config_self_sufficient = True
            has_defined_output_feature = True
        except Exception as e:
            logger.warning(f"Failed to auto-populate GeminiConfig output_features: {e}")

    # If features are not self-sufficient in cfg, and no external source is provided, it's an error.
    if not is_config_self_sufficient and ds_meta is None and env_cfg is None:
        # This warning is good, but the error should be more specific if output_features is the issue
        logger.warning(
            f"Policy config for '{cfg.type}' does not seem to have pre-defined input/output features, "
            "and neither dataset metadata nor a sim env was provided to infer them. "
            "This might lead to errors if the policy requires specific feature shapes. "
            "Attempting to proceed, but it's recommended to ensure 'output_features' (action) "
            "and 'input_features' (observation) are correctly set in the policy's configuration, "
            "or provide ds_meta/env_cfg."
        )
        raise ValueError(
            "Policy features (especially output_features defining action_space) are not adequately defined "
            "in the policy config, and neither dataset metadata nor a sim env was provided to infer them. "
            "Please ensure your policy config (e.g., GeminiConfig) initializes 'output_features' correctly with a valid shape."
        )

    # Early exit: if config self-sufficient and no ds_meta/env overrides, instantiate immediately
    if is_config_self_sufficient and ds_meta is None and env_cfg is None:
        policy_cls = get_policy_class(cfg.type)
        basic_kwargs = {"config": cfg}
        if cfg.pretrained_path:
            basic_kwargs["pretrained_name_or_path"] = cfg.pretrained_path
            policy = policy_cls.from_pretrained(**basic_kwargs)
        else:
            policy = policy_cls(**basic_kwargs)
        policy.to(cfg.device)
        assert isinstance(policy, nn.Module)
        return policy

    # Initialize policy_features from external source if provided
    policy_features_source = None
    if ds_meta is not None:
        policy_features_source = dataset_to_policy_features(ds_meta.features)
    elif env_cfg is not None:
        policy_features_source = env_to_policy_features(env_cfg)

    # If external features are provided, update the config from them.
    # Otherwise, the config is assumed to be self-sufficient or will be handled by from_pretrained.
    if policy_features_source is not None:
        # This part assumes cfg.from_policy_features exists and correctly merges/updates features.
        # It might be safer to directly set cfg.input_features and cfg.output_features here.
        cfg.output_features = {key: ft for key, ft in policy_features_source.items() if ft.type is FeatureType.ACTION}
        cfg.input_features = {key: ft for key, ft in policy_features_source.items() if key not in cfg.output_features}
        # Potentially call cfg.from_policy_features(policy_features_source) if it does more than just setting these two.
    else:
        # Only derive features from env if config still needs them and env_cfg provided
        if not is_config_self_sufficient and env_cfg is not None:
            if not cfg.pretrained_path:
                logging.warning(
                    "You are instantiating a policy from scratch and its features are parsed from an environment "
                    "rather than a dataset. Normalization modules inside the policy will have infinite values "
                    "by default without stats from a dataset."
                )
            features = env_to_policy_features(env_cfg)
            cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
            cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}

    # If config was already self-sufficient, we keep its existing features untouched.

    policy_cls = get_policy_class(cfg.type)

    # NOTE: Currently, if you try to run vqbet with mps backend, you'll get this error.
    # TODO(aliberts, rcadene): Implement a check_backend_compatibility in policies?
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    kwargs = {}
    if ds_meta is not None:
        # Parse feature information from the dataset to dimension the policy.
        features = dataset_to_policy_features(ds_meta.features)

        # Only forward dataset statistics if they are available (i.e., the
        # dataset already contains at least one episode).  When we are just
        # *recording* a brand-new dataset, `ds_meta.stats` will be empty and
        # passing it would overwrite the meaningful statistics that are stored
        # inside the pretrained checkpoint, leading to `inf` buffers and an
        # assertion error at inference time.
        stats_candidate = getattr(ds_meta, "stats", None)
        if stats_candidate:
            # Skip freshly-created datasets whose default stats still contain
            # infinities (they are placeholders until the first episode is
            # written).  Using them would zero-out the pretrained statistics
            # and trigger "mean is inf" assertions during inference.
            def _contains_inf(d):
                import torch

                for sub in d.values():
                    for tensor in sub.values():
                        if torch.isinf(tensor).any():
                            return True
                return False

            if not _contains_inf(stats_candidate):
                kwargs["dataset_stats"] = stats_candidate
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        features = env_to_policy_features(env_cfg)

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    # If the config already defines its I/O features and we have no dataset or env_cfg overrides,
    # we can directly instantiate the policy without any extra feature-parsing logic.
    if is_config_self_sufficient and ds_meta is None and env_cfg is None:
        policy_cls = get_policy_class(cfg.type)
        kwargs = {"config": cfg}
        if cfg.pretrained_path:
            kwargs["pretrained_name_or_path"] = cfg.pretrained_path
            policy = policy_cls.from_pretrained(**kwargs)
        else:
            policy = policy_cls(**kwargs)

        policy.to(cfg.device)
        assert isinstance(policy, nn.Module)
        return policy

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy
