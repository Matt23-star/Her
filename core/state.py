from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    # Short-term memory: current turn input and conversation history
    messages: List[Message] = field(default_factory=list)
    # Voice-related: original audio/transcribed text
    audio_input_path: Optional[str] = None
    stt_text: Optional[str] = None
    # RAG retrieved context
    retrieved_context: List[str] = field(default_factory=list)
    # Tool call traces
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Current turn model output
    response_text: Optional[str] = None
    # Whether to store current turn info into long-term memory
    should_write_memory: bool = False
    # Arbitrary extensions
    extra: Dict[str, Any] = field(default_factory=dict)

    def add_user_message(self, content: str, **meta: Any) -> None:
        self.messages.append(Message(role="user", content=content, meta=meta))

    def add_assistant_message(self, content: str, **meta: Any) -> None:
        self.messages.append(Message(role="assistant", content=content, meta=meta))

    def last_user_text(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

