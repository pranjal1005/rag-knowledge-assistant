"""
LLM abstraction over Anthropic and OpenAI, chosen at runtime via
config.settings.llm_provider so the rest of the app doesn't care which one
is behind it.

Design choice: prompt construction (`build_prompt`) is a pure function with
no network dependency, kept separate from `call_llm` (which does the actual
API call). That split is what makes this file testable without an API key --
see tests/test_prompt.py. It also makes prompt-injection-ish issues in
retrieved context easier to reason about in isolation from the network code.
"""
from src.chunking import Chunk
from src.config import settings

SYSTEM_PROMPT = (
    "You are a technical knowledge-base assistant. Answer the user's question "
    "using ONLY the numbered context passages provided below. "
    "Every claim in your answer must be traceable to a passage -- cite the "
    "passage number in square brackets, e.g. [2], right after the claim it "
    "supports. If the passages don't contain enough information to answer, "
    "say so explicitly instead of guessing. Do not use outside knowledge. "
    "Respond in plain text only -- no markdown formatting (no **, no #, "
    "no bullet-point asterisks). The web UI displaying your answer does not "
    "render markdown, so any markdown syntax would show up as literal "
    "characters to the user. Use plain sentences and paragraphs, or a "
    "numbered list written as '1.', '2.' if a list is genuinely clearer."
)


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Builds the user-turn content: numbered context passages + the question."""
    if not chunks:
        context_block = "(no relevant passages were found in the knowledge base)"
    else:
        context_block = "\n\n".join(
            f"[{i + 1}] (source: {c.source_title})\n{c.text}"
            for i, c in enumerate(chunks)
        )

    return (
        f"Context passages:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer, citing passage numbers in square brackets for every claim:"
    )


def call_llm(prompt: str) -> str:
    """Dispatches to the configured provider. Raises RuntimeError if no key is set."""
    provider = settings.llm_provider

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file."
            )
        return _call_anthropic(prompt)

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Add it to your .env file."
            )
        return _call_openai(prompt)

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set. "
                "Add it to your .env file."
            )
        return _call_gemini(prompt)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'anthropic', 'openai', or 'gemini')")


def _call_anthropic(prompt: str) -> str:
    import anthropic  # imported lazily so the package is only required if actually used

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_openai(prompt: str) -> str:
    from openai import OpenAI  # imported lazily, same reasoning as above

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _call_gemini(prompt: str) -> str:
    # Google's current SDK (google-genai, `from google import genai`) --
    # not the older, now-legacy `google-generativeai` package. Imported
    # lazily, same reasoning as the other two providers.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text or ""


def generate_answer(question: str, chunks: list[Chunk]) -> str:
    prompt = build_prompt(question, chunks)
    return call_llm(prompt)
