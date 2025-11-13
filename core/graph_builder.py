from __future__ import annotations

from typing import Dict, Any, List

from .state import AgentState
from .llm_manager import LLMManager
from .memory_manager import MemoryManager
from .reasoning import ToolRouter, run_tool
from .tools.web_search import WebSearchTool
from .tools.summarize import SummarizeTool
from .tools.emotion_detect import EmotionDetectTool


class DialogueGraph:
    def __init__(self, llm: LLMManager, memory: MemoryManager, prompts: Dict[str, str]) -> None:
        self.llm = llm
        self.memory = memory
        self.prompts = prompts
        self.router = ToolRouter()
        self.tools = {
            "web_search": WebSearchTool(),
            "summarize": SummarizeTool(),
            "emotion_detect": EmotionDetectTool(),
        }

    # [1] SpeechInput -> [2] STT(Whisper)
    def node_stt(self, state: AgentState) -> AgentState:
        # Placeholder: use stt_text if filled, otherwise use last user message
        if not state.stt_text:
            state.stt_text = state.last_user_text() or ""
        return state

    # [3] ShortTermMemoryUpdate
    def node_short_term_memory(self, state: AgentState) -> AgentState:
        if state.stt_text:
            state.add_user_message(state.stt_text, source="stt")
        return state

    # [4] ContextRetrieval(RAG)
    def node_rag(self, state: AgentState) -> AgentState:
        q = state.last_user_text() or ""
        state.retrieved_context = self.memory.search(q) if q else []
        return state

    # [5a] ToolSelection - Select the tool to use
    def node_tool_selection(self, state: AgentState) -> AgentState:
        user_text = state.last_user_text() or ""
        decision = self.router.select(user_text)
        state.extra["decision"] = decision
        return state

    # [5b] Direct LLM response (no tools needed)
    def node_llm_direct(self, state: AgentState) -> AgentState:
        system_prompt = self.prompts.get("system", "你是一个助理。")
        messages = [
            {"role": "system", "content": system_prompt},
        ] + [{"role": m.role, "content": m.content} for m in state.messages[-10:]]
        if state.retrieved_context:
            messages.append({"role": "system", "content": "相关上下文：\n" + "\n".join(state.retrieved_context)})
        reply = self.llm.chat(system_prompt, messages)
        state.response_text = reply
        state.add_assistant_message(reply, mode="llm")
        return state

    # [5c] WebSearch tool node
    def node_web_search(self, state: AgentState) -> AgentState:
        user_text = state.last_user_text() or ""
        tool_output, meta = run_tool("web_search", self.tools, user_text, state.retrieved_context)
        state.tool_calls.append(meta)
        state.extra["tool_output"] = tool_output
        return state

    # [5d] Summarize tool node
    def node_summarize(self, state: AgentState) -> AgentState:
        user_text = state.last_user_text() or ""
        tool_output, meta = run_tool("summarize", self.tools, user_text, state.retrieved_context)
        state.tool_calls.append(meta)
        state.extra["tool_output"] = tool_output
        return state

    # [5e] EmotionDetect tool node
    def node_emotion_detect(self, state: AgentState) -> AgentState:
        user_text = state.last_user_text() or ""
        tool_output, meta = run_tool("emotion_detect", self.tools, user_text, state.retrieved_context)
        state.tool_calls.append(meta)
        state.extra["tool_output"] = tool_output
        return state

    # [5f] Merge tool results and generate LLM response
    def node_merge_tool_result(self, state: AgentState) -> AgentState:
        user_text = state.last_user_text() or ""
        tool_name = state.extra.get("decision", "unknown")
        tool_output = state.extra.get("tool_output", "")
        system_prompt = self.prompts.get("system", "你是一个助理。")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "system", "content": f"工具[{tool_name}] 输出：\n{tool_output}"},
        ]
        reply = self.llm.chat(system_prompt, messages)
        state.response_text = reply
        state.add_assistant_message(reply, mode=tool_name)
        return state

    # [5] Backward compatibility: sequential reasoning execution (for fallback mode)
    def node_reasoning(self, state: AgentState) -> AgentState:
        state = self.node_tool_selection(state)
        decision = state.extra.get("decision", "llm")
        if decision == "llm":
            state = self.node_llm_direct(state)
        else:
            if decision == "web_search":
                state = self.node_web_search(state)
            elif decision == "summarize":
                state = self.node_summarize(state)
            elif decision == "emotion_detect":
                state = self.node_emotion_detect(state)
            state = self.node_merge_tool_result(state)
        return state

    # [6] LongTermMemoryStoreDecision
    def node_memory_decision_and_write(self, state: AgentState) -> AgentState:
        # Simple heuristic: write when response is long or contains preference keywords
        text = (state.last_user_text() or "") + "\n" + (state.response_text or "")
        should = len(text) > 80 or any(k in text for k in ["我喜欢", "我的目标", "我的生日", "我的城市"])
        state.should_write_memory = should
        if should:
            self.memory.add_memory(text, meta={"type": "dialogue"})
        return state

    # [7] TTS(voice output) -> No audio processing here, delegated to voice subsystem; placeholder return
    def node_tts(self, state: AgentState) -> AgentState:
        # Placeholder: no operation
        return state

    def run_turn(self, state: AgentState) -> AgentState:
        state = self.node_stt(state)
        state = self.node_short_term_memory(state)
        state = self.node_rag(state)
        state = self.node_reasoning(state)
        state = self.node_memory_decision_and_write(state)
        state = self.node_tts(state)
        return state


def build_graph(llm: LLMManager, memory: MemoryManager, prompts: Dict[str, str]) -> DialogueGraph:
    return DialogueGraph(llm=llm, memory=memory, prompts=prompts)

