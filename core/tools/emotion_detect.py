from __future__ import annotations


class EmotionDetectTool:
    def __init__(self) -> None:
        self.keywords = {
            "开心": "joy",
            "高兴": "joy",
            "难过": "sadness",
            "伤心": "sadness",
            "生气": "anger",
            "愤怒": "anger",
            "害怕": "fear",
            "担心": "fear",
            "平静": "neutral",
            "还好": "neutral",
        }

    def run(self, text: str) -> str:
        for k, v in self.keywords.items():
            if k in (text or ""):
                return f"检测到情绪：{v}"
        return "情绪倾向：neutral（占位）"

