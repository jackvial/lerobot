#!/usr/bin/env python

"""GeminiPolicy: a minimal LeRobot policy that delegates action selection to
Google Gemini through the `langchain_google_genai` wrapper.

The policy is *not* trainable.  It is intended for quick prototyping or manual
control experiments where a large-language model produces low-level joint
commands from a natural-language prompt and the current robot observation.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from typing import Any, List

import torch
from langchain_google_genai import ChatGoogleGenerativeAI

from lerobot.common.constants import OBS_ROBOT
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.gemini.configuration_gemini import GeminiConfig


class GeminiPolicy(PreTrainedPolicy):
    """Policy wrapper that queries Gemini for every *n_action_steps* environment steps."""

    config_class = GeminiConfig
    name = "gemini"

    def __init__(self, config: GeminiConfig, *args: Any, **kwargs: Any):
        super().__init__(config)
        self.config = config

        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise EnvironmentError(
                "Environment variable GOOGLE_API_KEY must be set to use GeminiPolicy."
            )

        self._llm = ChatGoogleGenerativeAI(
            model=config.model, temperature=config.temperature, api_key=api_key
        )

        self.reset()

    # ---------------------------------------------------------------------
    # PreTrainedPolicy interface
    # ---------------------------------------------------------------------
    def get_optim_params(self):  # noqa: D401  # type: ignore[override]
        # The policy has no trainable parameters but returning an empty list keeps
        # the training pipeline happy if someone mistakenly tries to train it.
        return []

    def reset(self):  # type: ignore[override]
        # Buffer of forthcoming actions produced by the latest LLM call.
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    # NOTE: We purposefully *do not* implement a usable training `forward` – this
    # policy is inference-only.
    def forward(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError("GeminiPolicy is not trainable.")

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:  # type: ignore[override]
        """Return a single *unnormalised* action for each env in the batch."""

        # If we still have actions left from the previous LLM invocation, just
        # pop and return the next one.
        if len(self._action_queue) > 0:
            return self._action_queue.popleft()

        # Otherwise, generate a fresh sequence from Gemini.
        self._enqueue_new_actions(batch)
        # There is now at least one action in the queue.
        return self._action_queue.popleft()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _enqueue_new_actions(self, batch: dict[str, Tensor]) -> None:
        """Query Gemini and fill `_action_queue` with a new sequence."""

        # We assume the *first* element in the vectorised batch represents the
        # real robot we are controlling.  Extract its current joint state if
        # available.
        obs_vector: Tensor | None = None
        if OBS_ROBOT in batch:
            obs_vector = batch[OBS_ROBOT][0]
        elif "agent_pos" in batch:  # fallback naming used in some envs
            obs_vector = batch["agent_pos"][0]

        obs_text = (
            f"Current joint positions: {obs_vector.tolist()}\n" if obs_vector is not None else ""
        )

        # Compose the full user prompt.
        action_dim = self.config.action_feature.shape[0] if self.config.action_feature else 0
        user_prompt = (
            f"{self.config.prompt}\n\n"
            f"Provide exactly {self.config.n_action_steps} actions as JSON. "
            f"Each action must be a list of {action_dim} floats in the range [-1, 1]. "
            "Return *only* the JSON list – no additional text."
        )

        messages = [
            {"role": "system", "content": obs_text + user_prompt},
        ]

        try:
            response = self._llm.invoke(messages).content  # type: ignore[call-arg]
        except Exception as e:  # pragma: no cover – network / quota errors.
            # If the LLM call fails we default to zeros to keep the robot safe.
            print(f"[GeminiPolicy] LLM call failed: {e}")
            actions = [torch.zeros(action_dim) for _ in range(self.config.n_action_steps)]
        else:
            actions = self._parse_actions_from_response(response, action_dim)

        # Pad / truncate so that we have exactly n_action_steps tensors.
        if len(actions) < self.config.n_action_steps:
            actions.extend([torch.zeros(action_dim) for _ in range(self.config.n_action_steps - len(actions))])
        elif len(actions) > self.config.n_action_steps:
            actions = actions[: self.config.n_action_steps]

        # Push into queue.
        for act in actions:
            self._action_queue.append(act)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_actions_from_response(response: str, action_dim: int) -> List[Tensor]:
        """Extract a list of tensors from the LLM raw response."""

        # naive – find first JSON array.
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            return [torch.zeros(action_dim)]  # fallback safe-action

        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return [torch.zeros(action_dim)]

        # Make sure we end up with *list[list[float]]*.
        if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], (int, float))):
            data = [data]  # single action flattened

        # Clip / pad to expected length.
        sequence: List[Tensor] = []
        for idx in range(len(data)):
            if idx < len(data) and isinstance(data[idx], list) and len(data[idx]) == action_dim:
                vec = torch.tensor(data[idx], dtype=torch.float32)
            else:
                vec = torch.zeros(action_dim)
            vec = torch.clamp(vec, -1.0, 1.0)
            sequence.append(vec)

        return sequence 