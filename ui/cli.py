from __future__ import annotations

import sys
from typing import Dict

from core.state import AgentState
from core.llm_manager import LLMManager
from core.memory_manager import MemoryManager
try:
    from core.langgraph_builder import build_graph  # Prefer LangGraph, will auto-fallback internally
except Exception:
    from core.graph_builder import build_graph


def load_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def run_cli(config: Dict) -> None:
    llm_conf = config.get("llm", {})
    rag_conf = config.get("rag", {})
    prompts_conf = config.get("prompts", {})

    system_prompt = load_text(prompts_conf.get("system_prompt", ""))
    memory_prompt = load_text(prompts_conf.get("memory_prompt", ""))

    llm = LLMManager(
        provider=llm_conf.get("provider", "openai"),
        model=llm_conf.get("model", "gpt-4o-mini"),
        temperature=float(llm_conf.get("temperature", 0.7)),
        api_key_env=llm_conf.get("api_key_env", "OPENAI_API_KEY"),
    )
    memory = MemoryManager(
        persist_path=rag_conf.get("persist_dir", "data/vector_store/memories.jsonl") + "/memories.jsonl"
        if rag_conf.get("persist_dir", "").endswith("/")
        else f"{rag_conf.get('persist_dir', 'data/vector_store')}/memories.jsonl",
        top_k=int(rag_conf.get("top_k", 4)),
    )

    graph = build_graph(
        llm=llm,
        memory=memory,
        prompts={"system": system_prompt, "memory": memory_prompt},
    )

    print("Her 风格对话（输入 'exit' 退出）")
    state = AgentState()
    while True:
        try:
            user_in = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见👋")
            break
        if user_in.lower() in {"exit", "quit"}:
            print("再见👋")
            break
        state.add_user_message(user_in, source="cli")
        state.stt_text = user_in  # In CLI mode, treat text input as STT output directly
        state = graph.run_turn(state)
        print(f"她: {state.response_text}")

