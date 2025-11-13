from __future__ import annotations


class SummarizeTool:
    def __init__(self) -> None:
        pass

    def run(self, text: str) -> str:
        # Minimal: take head and tail fragments
        text = text.strip()
        if not text:
            return "(无可总结内容)"
        head = text[:80]
        tail = text[-80:] if len(text) > 160 else ""
        return f"总结：{head}{' ... ' if tail else ''}{tail}"

