from __future__ import annotations

from typing import Optional


class STTEngine:
    def __init__(self, provider: str = "openai", model: str = "whisper-1") -> None:
        self.provider = provider
        self.model = model

    def transcribe(self, audio_path: str) -> Optional[str]:
        # Placeholder: returns None, indicating fallback to CLI text input
        return None

