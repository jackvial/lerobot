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
from typing import List

import torch
from torch import Tensor
import numpy as np

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.gemini.configuration_gemini import GeminiConfig

# Try to import the LangChain wrapper for Gemini.  We keep the import optional
# so that the rest of the codebase can run even if the dependency is absent.
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ModuleNotFoundError:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]


class GeminiPolicy(PreTrainedPolicy):
    """A very lightweight policy that delegates action generation to Gemini.

    At the first environment step (or whenever the internal action buffer is
    empty) the policy sends a prompt to Gemini and expects back *n_action_steps*
    joint-angle commands formatted as a list.  It then serves one time-step at a
    time from this buffer.
    """

    config_class = GeminiConfig
    name = "gemini"

    def __init__(self, config: GeminiConfig, *_, **__):  # noqa: D401
        super().__init__(config)
        self.config: GeminiConfig = config

        if ChatGoogleGenerativeAI is None:
            raise ImportError(
                "GeminiPolicy requires the `langchain_google_genai` package.  Install via `pip install langchain_google_genai`."
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set – cannot authenticate with Gemini.")

        # Initialise the LangChain chat wrapper.
        self.llm = ChatGoogleGenerativeAI(model=config.model_name, api_key=api_key)

        # Internal FIFO queue storing upcoming actions (shape: (action_dim,))
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)

    # ------------------------------------------------------------------
    # PreTrainedPolicy abstract-method implementations
    # ------------------------------------------------------------------

    def reset(self):  # noqa: D401
        self._action_queue.clear()

    def get_optim_params(self):  # noqa: D401
        # No trainable parameters
        return []

    def forward(self, batch):  # noqa: D401
        raise NotImplementedError("GeminiPolicy is for inference only – training is not supported.")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:  # noqa: D401
        """Return a single action for the current environment step."""

        # If we still have buffered actions, just pop and return.
        if self._action_queue:
            return self._action_queue.popleft()

        # Otherwise we need to query Gemini for the next *n_action_steps*.
        action_dim = self.config.action_feature.shape[0]
        response = self._query_gemini(action_dim)

        # Parse Gemini's response into a list[Tensor].  If parsing fails we fall
        # back to zeros for safety.
        actions = self._parse_actions(response, action_dim)
        if not actions:
            actions = [torch.zeros(action_dim)]

        # Push to queue and pop the first element to return.
        for act in actions:
            self._action_queue.append(act)
        return self._action_queue.popleft()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_gemini(self, action_dim: int) -> str:
        """Send a prompt to Gemini and return the raw string response."""

        system_prompt = (
            "You control a 6-DoF robot arm.  Respond ONLY with a JSON array of "
            f"{self.config.n_action_steps} arrays, each containing {action_dim} "
            "floating-point numbers that represent joint target angles in degrees."
        )
        user_prompt = self.config.prompt or "Provide joint targets."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = self.llm.invoke(messages)
        # Langchain returns an object with `.content` holding the assistant text.
        return getattr(result, "content", str(result))

    def _parse_actions(self, text: str, action_dim: int) -> List[Tensor]:
        """Extract a list of action tensors from Gemini's response string."""
        # First try JSON parsing
        try:
            data = json.loads(text)
            if (
                isinstance(data, list)
                and all(isinstance(elem, list) and len(elem) == action_dim for elem in data)
            ):
                return [torch.tensor(elem, dtype=torch.float32) for elem in data]
        except Exception:
            pass

        # Fallback: use regex to extract floats and chunk.
        floats = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)]
        if len(floats) >= action_dim:
            # Take up to n_action_steps groups
            actions = []
            idx = 0
            for _ in range(self.config.n_action_steps):
                if idx + action_dim > len(floats):
                    break
                actions.append(torch.tensor(floats[idx : idx + action_dim], dtype=torch.float32))
                idx += action_dim
            return actions
        return [] 