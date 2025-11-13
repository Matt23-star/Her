from __future__ import annotations

from typing import Dict, Any, List, Tuple


class ToolRouter:
    """Simple heuristic tool selection:
    - Select web_search/summarize/emotion_detect when keywords are present
    - Otherwise use direct LLM response
    """

    def __init__(self) -> None:
        self.rules = [
            ("搜索", "web_search"), ("查", "web_search"), ("新闻", "web_search"),
            ("总结", "summarize"), ("概括", "summarize"),
            ("心情", "emotion_detect"), ("情绪", "emotion_detect"),
        ]

    def select(self, user_text: str) -> str:
        text = user_text or ""
        for kw, tool in self.rules:
            if kw in text:
                return tool
        return "llm"


def run_tool(tool_name: str, tools: Dict[str, Any], query: str, context: List[str]) -> Tuple[str, Dict[str, Any]]:
    if tool_name == "web_search":
        result = tools["web_search"].run(query)
        return result, {"tool": tool_name}
    if tool_name == "summarize":
        result = tools["summarize"].run("\n".join(context) or query)
        return result, {"tool": tool_name}
    if tool_name == "emotion_detect":
        result = tools["emotion_detect"].run(query)
        return result, {"tool": tool_name}
    return "", {"tool": "none"}

