from __future__ import annotations

from typing import Optional


class TTSEngine:
    def __init__(self, provider: str = "openai", voice: str = "allison") -> None:
        self.provider = provider
        self.voice = voice

    def synthesize(self, text: str, out_path: Optional[str] = None) -> Optional[str]:
        # Placeholder: no audio generation, returns None
        return None

