"""Export LangGraph structure as Mermaid text or PNG."""

import argparse
from pathlib import Path

from shared.config import create_chat_model, load_settings
from langgraph_demo.main import build_graph

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the LangGraph chat demo")
    parser.add_argument("--png", action="store_true", help="Also write langgraph.png")
    args = parser.parse_args()

    settings, _, provider = load_settings()
    graph = build_graph(create_chat_model(settings))

    OUTPUT_DIR.mkdir(exist_ok=True)
    mermaid_path = OUTPUT_DIR / "langgraph.mmd"
    mermaid = graph.get_graph().draw_mermaid()
    mermaid_path.write_text(mermaid, encoding="utf-8")

    print(f"Provider: {provider.name} | Model: {settings.model}")
    print(mermaid)
    print(f"\nSaved: {mermaid_path}")

    if args.png:
        png_path = OUTPUT_DIR / "langgraph.png"
        png_path.write_bytes(graph.get_graph().draw_mermaid_png())
        print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
