"""Compiled graph export for LangGraph Studio (`langgraph dev`)."""

from shared.config import create_chat_model, load_settings
from langgraph_demo.main import build_graph

settings, _, _ = load_settings()
graph = build_graph(create_chat_model(settings), use_checkpointer=False)
