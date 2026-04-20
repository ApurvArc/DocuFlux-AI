import os
import re
from functools import lru_cache
from pathlib import Path

from chromadb import PersistentClient as ChromaClient
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


from core.config import (
    LM_STUDIO_BASE,
    MODEL,
    OPENAI_API_KEY,
    DB_NAME,
    KNOWLEDGE_BASE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_K,
    FINAL_K,
    MAX_CONTEXT_CHARS,
    MODEL_PROVIDERS,
)

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using the Context provided below.

IMPORTANT RULES:
1. Base your answer on the Context below. Summarize, explain, and elaborate on what the context says.
2. If the Context contains relevant information, USE IT to give a thorough answer. Do not ignore available context.
3. If the Context does NOT contain any information related to the question, respond ONLY with: "The provided documents don't cover this topic. Try uploading a relevant document or asking about something in your current documents."
4. NEVER make up facts, names, events, or details that are not in the Context.
5. When possible, mention the source document name.

Context:
{context}
"""

embedder = None
collection = None
_llm_cache: dict[str, ChatOpenAI] = {}


class Result(BaseModel):
    page_content: str
    metadata: dict


def get_embedder() -> SentenceTransformer:
    """Load the embedding model, downloading it on first run if needed."""
    global embedder
    if embedder is None:
        embedder = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            cache_folder=os.getenv("HF_HOME"),
        )
    return embedder


def get_collection():
    """Connect lazily so importing the module stays fast and offline-safe."""
    global collection
    if collection is None:
        chroma = ChromaClient(path=DB_NAME)
        collection = chroma.get_or_create_collection(COLLECTION_NAME)
    return collection


def get_arbitrary_collection(db_path: str):
    """Connect to any Chroma DB path without altering global state."""
    chroma = ChromaClient(path=db_path)
    return chroma.get_or_create_collection("session")


def get_llm(provider: str = "Local (LM Studio)") -> ChatOpenAI:
    """Return a cached LLM client for the given provider."""
    if provider not in _llm_cache:
        cfg = MODEL_PROVIDERS.get(provider, MODEL_PROVIDERS["Local (LM Studio)"])
        if not cfg["api_key"]:
            raise ValueError(f"API key for '{provider}' is not set in your .env file.")
        _llm_cache[provider] = ChatOpenAI(
            temperature=0,
            model_name=cfg["model"],
            openai_api_base=cfg["base_url"],
            openai_api_key=cfg["api_key"],
        )
    return _llm_cache[provider]


@lru_cache(maxsize=1)
def load_markdown_documents() -> list[Result]:
    """Load raw markdown files for the lexical fallback path."""
    documents = []
    for path in KNOWLEDGE_BASE.rglob("*.md"):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(
                Result(
                    page_content=text,
                    metadata={"source": str(path), "doc_type": path.parent.name},
                )
            )
    return documents


def fetch_keyword_context(question: str, limit: int = RETRIEVAL_K) -> list[Result]:
    """Fallback retrieval based on keyword overlap with raw markdown files."""
    terms = {t for t in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(t) > 2}
    if not terms:
        return []

    def make_snippet(text: str, source: str) -> str:
        lower_text = text.lower()
        positions = [lower_text.find(t) for t in terms if lower_text.find(t) >= 0]
        if not positions:
            snippet = text[:600]
        else:
            start = max(0, min(positions) - 220)
            end = min(len(text), max(positions) + 380)
            snippet = text[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
        title = Path(source).stem
        if title and title.lower() not in snippet.lower():
            snippet = f"# {title}\n\n{snippet}"
        return snippet

    scored_docs: list[tuple[int, Result]] = []
    for doc in load_markdown_documents():
        text = doc.page_content.lower()
        score = sum(1 for t in terms if t in text)
        if score:
            scored_docs.append((
                score,
                Result(
                    page_content=make_snippet(
                        doc.page_content,
                        doc.metadata.get("source", ""),
                    ),
                    metadata=doc.metadata,
                ),
            ))

    scored_docs.sort(key=lambda item: (
        -item[0],
        len(item[1].page_content),
        item[1].metadata.get("source", ""),
    ))
    return [doc for _, doc in scored_docs[:limit]]


def fetch_context_unranked(question: str, session_db_path: str | None = None) -> list[Result]:
    """Embed question and retrieve top-k chunks from Chroma."""
    coll = get_arbitrary_collection(session_db_path) if session_db_path else get_collection()

    try:
        vector = get_embedder().encode([question]).tolist()[0]
        results = coll.query(
            query_embeddings=[vector], 
            n_results=RETRIEVAL_K,
            include=["documents", "metadatas", "distances"]
        )
    except Exception:
        return []

    chunks = []
    if results and "documents" in results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results.get("distances", [[0]*len(docs)])[0]
        
        for doc, meta, dist in zip(docs, metas, dists):
            # Drop vectors that are completely unrelated (L2 distance > 1.4 is usually weak for all-MiniLM)
            if dist < 1.4:
                chunks.append(Result(page_content=doc, metadata=meta or {}))
    return chunks


def merge_chunks(chunks1: list[Result], chunks2: list[Result]) -> list[Result]:
    """Deduplicate and merge two result lists."""
    merged = chunks1[:]
    existing = {c.page_content for c in chunks1}
    for c in chunks2:
        if c.page_content not in existing:
            merged.append(c)
    return merged


def rewrite_query(question: str, provider: str = "Local (LM Studio)") -> str:
    """Rewrite the user question into a tighter KB search query."""
    prompt = (
        "You are searching a knowledge base.\n"
        "Rewrite the following question into a short, precise search query (one sentence max).\n"
        "Respond ONLY with the search query, nothing else.\n\n"
        f"Question: {question}"
    )
    try:
        response = get_llm(provider).invoke([SystemMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        return question


def rerank(question: str, chunks: list[Result]) -> list[Result]:
    """Rerank chunks deterministically using keyword overlap."""
    if len(chunks) <= 1:
        return chunks

    terms = {t for t in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(t) > 2}

    def score(chunk: Result) -> tuple[int, int, str]:
        text = chunk.page_content.lower()
        source = chunk.metadata.get("source", "").lower()
        return (
            sum(1 for t in terms if t in text),
            sum(1 for t in terms if t in source),
            chunk.metadata.get("source", ""),
        )

    return sorted(chunks, key=score, reverse=True)


def fallback_web_search(query: str) -> list[Result]:
    """Search DuckDuckGo when local context is missing."""
    try:
        from ddgs import DDGS as DDGS_NEW
        with DDGS_NEW() as ddgs:
            results = ddgs.text(query, region="us-en", max_results=3)
        chunks = []
        for r in (results or []):
            chunks.append(Result(
                page_content=r.get("body", ""),
                metadata={"source": r.get("href", "Web Search"), "doc_type": "web"}
            ))
        return chunks
    except Exception as e:
        print(f"Web search failed: {e}")
        return []


def fetch_context(question: str, session_db_path: str | None = None, provider: str = "Local (LM Studio)") -> list[Result]:
    """Full RAG retrieval: semantic + optional lexical fallback, then rerank."""
    rewritten = rewrite_query(question, provider)

    chunks1 = fetch_context_unranked(question, session_db_path)
    chunks2 = fetch_context_unranked(rewritten, session_db_path)
    merged = merge_chunks(chunks1, chunks2)

    if not session_db_path:
        lexical = fetch_keyword_context(question)
        merged = merge_chunks(merged, lexical)

    reranked = rerank(question, merged)[:FINAL_K]
    
    # AGENTIC FALLBACK: If nothing relevant was found locally, search the web!
    if not reranked:
        print("No local context found. Falling back to web search...")
        return fallback_web_search(question)
        
    return reranked


def build_context(chunks: list[Result], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Keep the prompt small enough for local models with tight context limits."""
    parts = []
    used = 0
    for chunk in chunks:
        snippet = f"[{chunk.metadata.get('source', 'unknown')}]\n{chunk.page_content}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining].rsplit(" ", 1)[0].rstrip() + "..."
        parts.append(snippet)
        used += len(snippet) + 2
    return "\n\n".join(parts)


def classify_input(text: str, provider: str = "Local (LM Studio)") -> str:
    """
    Decide whether the input is document context to store or a question to answer.
    Returns: 'context' | 'query'
    """
    stripped = text.strip()

    # Fast heuristics — no LLM round-trip needed
    if len(stripped) < 100:
        return "query"
    if len(stripped) > 500:
        return "context"
    if stripped.endswith("?"):
        return "query"

    # LLM classification for ambiguous medium-length text (100–500 chars)
    prompt = (
        "Classify the following user input for a document Q&A system.\n"
        "Reply with ONLY one word: \"context\" if it is a chunk of document text "
        "or factual information meant to be stored, or \"query\" if it is a question, "
        "command, or conversational message.\n\n"
        f"Input: {stripped[:400]}"
    )
    try:
        response = get_llm(provider).invoke([SystemMessage(content=prompt)])
        result = response.content.strip().lower()
        return "context" if "context" in result else "query"
    except Exception:
        question_starts = {
            "what", "why", "how", "when", "where", "who", "is", "are",
            "can", "could", "would", "should", "does", "do", "tell",
            "explain", "describe", "list", "give", "show", "find", "get",
        }
        first_word = stripped.lower().split()[0] if stripped else ""
        return "query" if first_word in question_starts else "context"


def answer_question(
    question: str,
    history: list[dict] | None = None,
    session_db_path: str | None = None,
    provider: str = "Local (LM Studio)",
) -> tuple[str, list]:
    """Answer a question using the RAG pipeline. Returns (answer, context_docs)."""
    if history is None:
        history = []

    chunks = fetch_context(question, session_db_path, provider=provider)
    context = build_context(chunks)
    system_prompt = SYSTEM_PROMPT.format(context=context)

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))

    try:
        response = get_llm(provider).invoke(messages)
        answer_text = response.content
    except Exception as e:
        err_str = str(e)
        # Detect quota / rate-limit errors (HTTP 429)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            answer_text = (
                f"**Rate limit reached for '{provider}'.**\n\n"
                "Your free-tier quota is exhausted for this model. "
                "Please switch to a different provider using the dropdown above, "
                "or wait for your quota to reset (usually within a few minutes or at midnight).\n\n"
                "---\n*Here is the most relevant context found in the meantime:*\n\n"
                f"{context}"
            )
        else:
            answer_text = (
                f"**Could not reach '{provider}'** — {err_str[:200]}\n\n"
                "Please check your API key or try a different provider.\n\n"
                "---\n*Here is the most relevant context found:*\n\n"
                f"{context}"
            )

    docs = [Document(page_content=c.page_content, metadata=c.metadata) for c in chunks]
    return answer_text, docs
