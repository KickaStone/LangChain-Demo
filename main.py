"""Simple LangChain assistant chat demo."""

import os
import sys

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely in the same language as the user."


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY is not set.\n"
            "Set it in your environment or create a .env file (see .env.example).",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def create_chat_chain(model_name: str):
    llm = ChatOpenAI(
        model=model_name,
        api_key=get_api_key(),
        temperature=0.7,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    history = InMemoryChatMessageHistory()

    def get_session_history(_session_id: str) -> InMemoryChatMessageHistory:
        return history

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def main() -> None:
    load_dotenv()
    model_name = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    chat = create_chat_chain(model_name)
    config = {"configurable": {"session_id": "demo"}}

    print(f"LangChain Demo (model: {model_name})")
    print("Type your question, or 'exit' / 'quit' to leave.\n")

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

        response = chat.invoke({"input": user_input}, config=config)
        print(f"Assistant: {response.content}\n")


if __name__ == "__main__":
    main()
