"""
Terminal REPL for asking the knowledge base questions.
Usage: python -m src.cli
"""
from src.rag_pipeline import answer_question
from src.vector_store import count


def main() -> None:
    n = count()
    if n == 0:
        print(
            "The vector store is empty. Build the index first:\n"
            "  python -m src.build_index --source local\n"
        )
        return

    print(f"Knowledge base ready ({n} chunks indexed). Ask a question, or 'quit' to exit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        result = answer_question(question)
        print(f"\n{result.text}\n")
        print("Sources:")
        for s in result.sources:
            print(f"  - {s.source_title} ({s.source_id}), distance={s.distance:.3f}")
        print()


if __name__ == "__main__":
    main()
