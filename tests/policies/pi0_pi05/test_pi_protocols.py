#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Tests that flat Pi model classes produce correct state_dict keys and satisfy Protocol contracts."""

import pytest

pytest.importorskip("transformers")

from transformers.models.auto import CONFIG_MAPPING  # noqa: E402

from lerobot.policies.pi_gemma import (  # noqa: E402
    PiDecoderModelProto,
    PiGemmaForCausalLM,
    PiGemmaModel,
    PiVLM,
)


def _make_vlm_config(use_adarms: bool = False, hidden_size: int = 128, depth: int = 2):
    """Build a minimal PaliGemma config for testing (small dims for speed)."""
    config = CONFIG_MAPPING["paligemma"]()
    config._vocab_size = 256  # noqa: SLF001
    config.image_token_index = 256
    config.text_config.hidden_size = hidden_size
    config.text_config.intermediate_size = hidden_size * 4
    config.text_config.num_attention_heads = 2
    config.text_config.head_dim = hidden_size // 2
    config.text_config.num_hidden_layers = depth
    config.text_config.num_key_value_heads = 1
    config.text_config.hidden_activation = "gelu_pytorch_tanh"
    config.text_config.dtype = "float32"
    config.text_config.vocab_size = 256
    config.text_config.use_adarms = use_adarms
    config.text_config.adarms_cond_dim = hidden_size if use_adarms else None
    config.vision_config.intermediate_size = 128
    config.vision_config.projection_dim = hidden_size
    config.vision_config.projector_hidden_act = "gelu_fast"
    config.vision_config.dtype = "float32"
    config.vision_config.image_size = 32
    config.vision_config.patch_size = 16
    config.vision_config.num_channels = 3
    config.vision_config.num_hidden_layers = 1
    config.vision_config.num_attention_heads = 2
    config.vision_config.hidden_size = 64
    return config


def _make_expert_config(use_adarms: bool = False, hidden_size: int = 128, depth: int = 2):
    """Build a minimal Gemma expert config for testing."""
    config = CONFIG_MAPPING["gemma"](
        head_dim=hidden_size // 2,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=2,
        num_hidden_layers=depth,
        num_key_value_heads=1,
        vocab_size=256,
        hidden_activation="gelu_pytorch_tanh",
        dtype="float32",
        use_adarms=use_adarms,
        adarms_cond_dim=hidden_size if use_adarms else None,
    )
    return config


# ---- Key path assertions ----

REQUIRED_VLM_KEY_PATTERNS = [
    "model.vision_tower.",
    "model.multi_modal_projector.",
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.self_attn.k_proj.weight",
    "model.language_model.layers.0.self_attn.v_proj.weight",
    "model.language_model.layers.0.self_attn.o_proj.weight",
    "model.language_model.layers.0.mlp.",
    "model.language_model.layers.0.input_layernorm.",
    "model.language_model.layers.0.post_attention_layernorm.",
    "model.language_model.norm.",
    "model.language_model.rotary_emb.",
]

REQUIRED_EXPERT_KEY_PATTERNS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.",
    "model.layers.0.input_layernorm.",
    "model.layers.0.post_attention_layernorm.",
    "model.norm.",
    "model.rotary_emb.",
    "model.embed_tokens.weight",
]


def _assert_key_patterns_present(keys: set[str], patterns: list[str], label: str):
    for pattern in patterns:
        if not any(pattern in k for k in keys):
            pytest.fail(f"[{label}] No key matching pattern '{pattern}' found in state_dict keys")


# ---- Tests ----


def test_pi0_vlm_state_dict_keys():
    """PI0 VLM (use_adarms=False) produces expected state_dict key paths."""
    config = _make_vlm_config(use_adarms=False)
    vlm = PiVLM(config=config)
    keys = set(vlm.state_dict().keys())
    _assert_key_patterns_present(keys, REQUIRED_VLM_KEY_PATTERNS, "PI0 VLM")


def test_pi0_expert_state_dict_keys():
    """PI0 expert (use_adarms=False) produces expected state_dict key paths."""
    config = _make_expert_config(use_adarms=False)
    expert = PiGemmaForCausalLM(config=config)
    keys = set(expert.state_dict().keys())
    _assert_key_patterns_present(keys, REQUIRED_EXPERT_KEY_PATTERNS, "PI0 Expert")


def test_pi05_vlm_state_dict_keys():
    """PI05 VLM (use_adarms=False on VLM) produces expected state_dict key paths."""
    config = _make_vlm_config(use_adarms=False)
    vlm = PiVLM(config=config)
    keys = set(vlm.state_dict().keys())
    _assert_key_patterns_present(keys, REQUIRED_VLM_KEY_PATTERNS, "PI05 VLM")


def test_pi05_expert_state_dict_keys_adarms():
    """PI05 expert with AdaRMS (use_adarms=True) produces expected state_dict key paths."""
    config = _make_expert_config(use_adarms=True)
    expert = PiGemmaForCausalLM(config=config)
    keys = set(expert.state_dict().keys())
    _assert_key_patterns_present(keys, REQUIRED_EXPERT_KEY_PATTERNS, "PI05 Expert")
    assert any("input_layernorm.dense." in k for k in keys), "AdaRMS expert should have layernorm dense"


def test_pi0fast_vlm_state_dict_keys():
    """PI0Fast VLM produces expected state_dict key paths."""
    config = _make_vlm_config(use_adarms=False)
    vlm = PiVLM(config=config)
    keys = set(vlm.state_dict().keys())
    _assert_key_patterns_present(keys, REQUIRED_VLM_KEY_PATTERNS, "PI0Fast VLM")


def test_pi0fast_vlm_adarms_reassignment():
    """PI0Fast AdaRMS reassignment produces expected state_dict key paths."""
    config = _make_vlm_config(use_adarms=True)
    vlm = PiVLM(config=config)
    text_config = vlm.config.text_config
    vlm.model.language_model = PiGemmaModel(text_config)
    keys = set(vlm.state_dict().keys())
    _assert_key_patterns_present(keys, REQUIRED_VLM_KEY_PATTERNS, "PI0Fast AdaRMS VLM")


def test_protocol_conformance_language_model():
    """PiGemmaModel satisfies PiDecoderModelProto."""
    config = _make_expert_config(use_adarms=False)
    model = PiGemmaModel(config)
    assert isinstance(model, PiDecoderModelProto), (
        f"PiGemmaModel should satisfy PiDecoderModelProto, got {type(model)}"
    )


def test_protocol_conformance_vlm_language_model():
    """VLM's inner language_model satisfies PiDecoderModelProto."""
    config = _make_vlm_config(use_adarms=False)
    vlm = PiVLM(config=config)
    assert isinstance(vlm.model.language_model, PiDecoderModelProto), (
        "VLM language_model should satisfy PiDecoderModelProto"
    )


def test_protocol_conformance_expert():
    """Expert's inner model satisfies PiDecoderModelProto."""
    config = _make_expert_config(use_adarms=False)
    expert = PiGemmaForCausalLM(config=config)
    assert isinstance(expert.model, PiDecoderModelProto), (
        "Expert model should satisfy PiDecoderModelProto"
    )


def test_no_duplicate_allocation():
    """Verify flat classes don't create throwaway modules (regression guard)."""
    import gc

    gc.collect()
    config = _make_vlm_config(use_adarms=False)
    vlm = PiVLM(config=config)
    # PiVLM should have exactly one vision_tower, one projector, one language_model
    assert hasattr(vlm.model, "vision_tower")
    assert hasattr(vlm.model, "multi_modal_projector")
    assert hasattr(vlm.model, "language_model")
    # language_model should be PiGemmaModel (flat), not GemmaModel
    assert type(vlm.model.language_model).__name__ == "PiGemmaModel"
