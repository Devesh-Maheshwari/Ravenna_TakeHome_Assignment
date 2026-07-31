"""Chat-model factory.

The single place in the codebase that names a provider. Nodes take a
`BaseChatModel` and never import `langchain_openai`, so switching providers is a
change to this file plus an env var — the orchestrator, tools, and prompts are
untouched.
"""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from support_agent.config import Settings


def build_chat_model(settings: Settings, *, tools: list | None = None) -> BaseChatModel:
    """Construct the chat model, optionally with tools bound.

    A bound model emits structured tool calls consumed by either the live bounded
    orchestrator or the optional LangGraph reference worker.
    """
    model = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    return model.bind_tools(tools) if tools else model


def build_embeddings(settings: Settings):
    """Embedding model for knowledge-base indexing and query encoding.

    Used by `scripts/seed_db.py` to index the 25 articles once, and by the
    knowledge-base tool to encode each incoming query.
    """
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
