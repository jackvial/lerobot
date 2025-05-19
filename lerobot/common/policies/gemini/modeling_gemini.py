#!/usr/bin/env python

"""GeminiPolicy: a minimal LeRobot policy that delegates action selection to
Google Gemini through the `langchain_google_genai` wrapper.

The policy is *not* trainable.  It is intended for quick prototyping or manual
control experiments where a large-language model produces low-level joint
commands from a natural-language prompt and the current robot observation.
"""

from __future__ import annotations

import json, os, re, base64, io
from collections import deque
from typing import List

import torch
from torch import Tensor
import numpy as np
from PIL import Image

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
            act = self._action_queue.popleft()
            # guarantee (batch, action_dim)
            if act.ndim == 1:
                act = act.unsqueeze(0)
            return act

        # Otherwise we need to query Gemini for the next *n_action_steps*.
        action_dim = self.config.action_feature.shape[0]

        blocks: list[dict] = [{"type": "text", "text": self.config.prompt}]

        if self.config.include_state and "observation.state" in batch:
            joints = batch["observation.state"][0].tolist()
            blocks.append({"type": "text", "text": f"Current joint angles (deg): {joints}"})
        
        # For sim
        if self.config.include_image and "observation.image" in batch:
            img = batch["observation.image"][0]  # (C,H,W) float32
            blocks.append({"type": "image_url", "image_url": self._img_to_url(img)})

        if self.config.include_image and "observation.images.top" in batch:
            img = batch["observation.images.top"][0]  # (C,H,W) float32
            blocks.append({"type": "image_url", "image_url": self._img_to_url(img)})

        # store for _query_gemini
        self._extra_blocks = blocks

        response = self._query_gemini(action_dim)

        # Parse Gemini's response into a list[Tensor].  If parsing fails we fall
        # back to zeros for safety.
        actions = self._parse_actions(response, action_dim)
        if not actions:
            actions = [torch.zeros(action_dim)]

        # Push to queue and pop the first element to return.
        for act in actions:
            self._action_queue.append(act)

        act = self._action_queue.popleft()
        if act.ndim == 1:
            act = act.unsqueeze(0)
        return act

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_gemini(self, action_dim: int) -> str:
        """Send a prompt to Gemini and return the raw string response."""

        w, h = self.config.image_resize  # width, height from config
        system_prompt = (
            "You are controlling a puck-shaped agent in the MuJoCo Push environment (`PushCubeLoop-v0`). "
            "The goal is to push a grey T so that its centre fully overlaps the green T target. "
            f"Respond ONLY with a JSON array containing {self.config.n_action_steps} sub-arrays, each with {action_dim} floating-point numbers. ",
            "start by moving down and to the right",
            "always try moving towards the grey T",
            "if you get stuck, move diagonally towards the center of the arena"
        )

        # `self._extra_blocks` is prepared in select_action
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._extra_blocks},
        ]

        result = self.llm.invoke(messages)
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

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _img_to_url(self, img: Tensor) -> str:
        """Convert CHW float32 image to data URL JPEG."""
        img_np = (img.clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy())
        pil = Image.fromarray(img_np)
        pil = pil.resize(self.config.image_resize, Image.BILINEAR)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=75)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}" 