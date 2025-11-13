from __future__ import annotations

from typing import Dict, Any

try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception:  # pragma: no cover
    StateGraph = None  # type: ignore
    END = None  # type: ignore

from .state import AgentState
from .llm_manager import LLMManager
from .memory_manager import MemoryManager
from .graph_builder import DialogueGraph


class LangGraphRunner:
    def __init__(self, app: Any) -> None:
        self.app = app

    def run_turn(self, state: AgentState) -> AgentState:
        # LangGraph's invoke returns the merged state; return it directly
        return self.app.invoke(state)


def route_after_tool_selection(state: AgentState) -> str:
    """Conditional routing function: route to different nodes based on tool selection decision"""
    decision = state.extra.get("decision", "llm")
    return decision


def build_graph(llm: LLMManager, memory: MemoryManager, prompts: Dict[str, str]):
    # Fallback to sequential DialogueGraph if LangGraph is not installed
    if StateGraph is None:
        return DialogueGraph(llm=llm, memory=memory, prompts=prompts)

    # Use LangGraph, wrap node functions as graph nodes
    seq = DialogueGraph(llm=llm, memory=memory, prompts=prompts)

    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("stt", seq.node_stt)
    graph.add_node("short_term", seq.node_short_term_memory)
    graph.add_node("rag", seq.node_rag)
    graph.add_node("tool_selection", seq.node_tool_selection)
    graph.add_node("llm_direct", seq.node_llm_direct)
    graph.add_node("web_search", seq.node_web_search)
    graph.add_node("summarize", seq.node_summarize)
    graph.add_node("emotion_detect", seq.node_emotion_detect)
    graph.add_node("merge_tool_result", seq.node_merge_tool_result)
    graph.add_node("memory", seq.node_memory_decision_and_write)
    graph.add_node("tts", seq.node_tts)

    # Set entry point
    graph.set_entry_point("stt")

    # Linear flow: stt -> short_term -> rag -> tool_selection
    graph.add_edge("stt", "short_term")
    graph.add_edge("short_term", "rag")
    graph.add_edge("rag", "tool_selection")

    # Conditional routing: tool_selection -> route to different nodes based on decision
    graph.add_conditional_edges(
        "tool_selection",
        route_after_tool_selection,
        {
            "llm": "llm_direct",
            "web_search": "web_search",
            "summarize": "summarize",
            "emotion_detect": "emotion_detect",
        },
    )

    # All tool nodes route to merge_tool_result after completion
    graph.add_edge("web_search", "merge_tool_result")
    graph.add_edge("summarize", "merge_tool_result")
    graph.add_edge("emotion_detect", "merge_tool_result")

    # Both llm_direct and merge_tool_result route to memory
    graph.add_edge("llm_direct", "memory")
    graph.add_edge("merge_tool_result", "memory")

    # Finally to tts and end
    graph.add_edge("memory", "tts")
    graph.add_edge("tts", END)

    app = graph.compile()
    return LangGraphRunner(app)

