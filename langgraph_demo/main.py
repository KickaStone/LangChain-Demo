"""LangGraph assistant chat demo with checkpointed conversation memory."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from shared.config import LLMSettings, build_settings, create_chat_model, load_settings
from shared.providers import Provider, get_provider, list_provider_ids

SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely in the same language as the user."
THREAD_ID = "demo"


def build_graph(llm, *, use_checkpointer: bool = True):
    """Build the chat graph.

    CLI mode uses an in-memory checkpointer. LangGraph Studio manages
    persistence itself, so pass use_checkpointer=False there.
    """
    def chatbot(state: MessagesState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = llm.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    if use_checkpointer:
        return builder.compile(checkpointer=InMemorySaver())
    return builder.compile()


class GraphSession:
    def __init__(
        self,
        settings: LLMSettings,
        available_models: list[str],
        provider: Provider,
    ):
        self.settings = settings
        self.available_models = available_models
        self.provider = provider
        self.llm = create_chat_model(settings)
        self.graph = build_graph(self.llm)
        self.config = {"configurable": {"thread_id": THREAD_ID}}

    def rebuild_graph(self) -> None:
        self.llm = create_chat_model(self.settings)
        self.graph = build_graph(self.llm)

    def switch_model(self, model_name: str) -> str:
        if model_name not in self.available_models:
            return f"Unknown model: {model_name}. Available: {', '.join(self.available_models)}"
        if model_name == self.settings.model:
            return f"Already using model: {model_name}"
        self.settings.model = model_name
        self.rebuild_graph()
        return f"Switched to model: {model_name}"

    def switch_provider(self, provider_id: str) -> str:
        try:
            provider = get_provider(provider_id)
        except ValueError as exc:
            return str(exc)

        if provider.id == self.provider.id:
            return f"Already using provider: {provider.name} ({provider.id})"

        settings, available_models = build_settings(provider)
        self.provider = provider
        self.settings = settings
        self.available_models = available_models
        self.rebuild_graph()
        return (
            f"Switched to provider: {provider.name} ({provider.id})\n"
            f"Endpoint: {self.settings.base_url}\n"
            f"Model: {self.settings.model}"
        )

    def chat(self, user_input: str) -> str:
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=self.config,
        )
        return result["messages"][-1].content


def format_providers(session: GraphSession) -> str:
    lines = [f"Current provider: {session.provider.name} ({session.provider.id})", ""]
    for provider_id in list_provider_ids():
        provider = get_provider(provider_id)
        marker = " *" if provider_id == session.provider.id else ""
        lines.append(f"  {provider.id:<12} {provider.name}{marker}")
    lines.extend(["", "Switch with: /provider <id>"])
    return "\n".join(lines)


def handle_command(session: GraphSession, user_input: str) -> str | None:
    if not user_input.startswith("/"):
        return None

    parts = user_input.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if command in {"/models", "/model"} and not arg:
        lines = [
            f"Provider: {session.provider.name} ({session.provider.id})",
            f"Current model: {session.settings.model}",
            f"Endpoint: {session.settings.base_url}",
            f"Thread: {THREAD_ID}",
            "Available models:",
            *[f"  - {name}" for name in session.available_models],
            "",
            "Switch with: /model <name>",
        ]
        return "\n".join(lines)

    if command == "/model" and arg:
        return session.switch_model(arg)

    if command in {"/providers", "/provider"} and not arg:
        return format_providers(session)

    if command == "/provider" and arg:
        return session.switch_provider(arg)

    if command == "/help":
        return (
            "Commands:\n"
            "  /providers         List API providers\n"
            "  /provider <id>     Switch provider\n"
            "  /models            List available models\n"
            "  /model <name>      Switch model\n"
            "  /help              Show this help\n"
            "  exit               Quit"
        )

    return f"Unknown command: {command}. Type /help for commands."


def main() -> None:
    settings, available_models, provider = load_settings()
    session = GraphSession(settings, available_models, provider)

    print(f"LangGraph Demo | {provider.name} ({provider.id}) | model: {session.settings.model}")
    print(f"Endpoint: {session.settings.base_url}")
    print("Type /help for commands, or 'exit' / 'quit' to leave.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        command_response = handle_command(session, user_input)
        if command_response is not None:
            print(f"{command_response}\n")
            continue

        print(f"Assistant: {session.chat(user_input)}\n")


if __name__ == "__main__":
    main()
