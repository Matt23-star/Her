from __future__ import annotations

import os
from typing import List, Dict, Any


class LLMManager:
    """Multimodal LLM wrapper (minimal runnable placeholder)"""

    def __init__(self, provider: str, model: str, temperature: float = 0.7, api_key_env: str = "OPENAI_API_KEY") -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv(api_key_env, "")

    def chat(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """
        Minimal placeholder: local response concatenation for running without API keys.
        Can be replaced with OpenAI/other SDKs for real integration.
        """
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"（占位回复）我已理解：{last_user[:200]}"

    def summarize_for_memory(self, text: str, memory_prompt: str) -> List[str]:
        # Simple sentence splitting placeholder
        points = [p.strip() for p in text.split("。") if p.strip()]
        return points[:5]

