import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from chromadb import PersistentClient as ChromaClient
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


from core.config import (
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
3. NEVER make up facts, names, events, or details that are not in the Context.
4. When possible, mention the source document name.

Context:
{context}
"""

WEB_FALLBACK_PROMPT = """You are a helpful assistant. The user's question was NOT found in the provided documents, so the context below was retrieved from a live web search.

IMPORTANT RULES:
1. ALWAYS begin your answer with this exact disclaimer on its own line:
   > [Web Search] This topic wasn't found in your provided documents. The following answer is based on a live web search.
2. Answer ONLY using the web search context below. Do not add facts from your training data.
3. If the web context below is clearly irrelevant or insufficient to answer the question, respond with:
   > [Web Search] This topic wasn't found in your provided documents, and the web search also didn't return a reliable answer. Try uploading a relevant document.
4. NEVER fabricate details not present in the context.
5. Cite the source URL at the end of your answer when available.

Web Search Context:
{context}
"""

# Common English words excluded from keyword overlap checks.
# Without this, words like "what", "how", "the" match any document text
# and produce false positives that silently block the web search fallback.
_STOP_WORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "the", "and", "but", "for", "are", "was", "were", "has", "have", "had",
    "not", "can", "could", "would", "should", "will", "shall", "may", "might",
    "does", "did", "its", "this", "that", "these", "those", "with", "from",
    "into", "about", "tell", "give", "show", "find", "get", "just", "also",
    "explain", "describe", "list",
    # 2-letter additions to safely support len > 1 acronyms (e.g. OS, DB, AI)
    "is", "on", "in", "to", "of", "it", "as", "be", "or", "at", "by", 
    "an", "am", "do", "go", "he", "me", "my", "no", "so", "up", "us", "we", "if"
}

WEB_SEARCH_K = 5

embedder = None
collection = None
_llm_cache: dict[str, ChatOpenAI] = {}
_chroma_cache = {}


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
    if db_path not in _chroma_cache:
        _chroma_cache[db_path] = ChromaClient(path=db_path)
    return _chroma_cache[db_path].get_or_create_collection("session")


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
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

    chunks = []
    if results and "documents" in results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results.get("distances", [[0]*len(docs)])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            # Relaxed threshold (1.6 instead of 1.2) to tolerate typos 
            # and short queries while still filtering out complete noise.
            if dist < 1.6:
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


# Words that signal a question depends on prior context
_FOLLOWUP_SIGNALS = {
    "it", "its", "that", "this", "those", "these", "they", "their",
    "more", "else", "also", "too", "above", "mentioned", "said",
    "same", "such", "previous", "earlier",
}


def condense_question(question: str, history: list[dict], provider: str) -> str:
    """
    If the question looks like a follow-up (uses pronouns or vague references),
    rewrite it as a fully self-contained standalone query using conversation history.
    Falls back to the original question on any error.
    """
    if not history:
        return question

    words = set(question.lower().split())
    is_followup = bool(words & _FOLLOWUP_SIGNALS) or len(question.split()) <= 5
    if not is_followup:
        return question

    # Use the last 2 turns (4 messages) for context
    recent = [
        m for m in history[-4:]
        if isinstance(m.get("content"), str)
    ]
    if not recent:
        return question

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}" for m in recent
    )
    prompt = (
        "Given the conversation history below and a follow-up question, "
        "rewrite the follow-up as a fully self-contained standalone question. "
        "Resolve all pronouns and references using the history context. "
        "Respond ONLY with the rewritten question, nothing else.\n\n"
        f"Conversation:\n{history_text}\n\n"
        f"Follow-up: {question}\n\n"
        "Standalone question:"
    )
    try:
        response = get_llm(provider).invoke([SystemMessage(content=prompt)])
        condensed = response.content.strip()
        return condensed if condensed else question
    except Exception:
        return question


def rerank(question: str, chunks: list[Result]) -> list[Result]:
    """Rerank chunks deterministically using keyword overlap."""
    if len(chunks) <= 1:
        return chunks

    terms = {t for t in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(t) > 1 and t not in _STOP_WORDS}
    term_patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]

    def score(chunk: Result) -> tuple[int, int, str]:
        text = chunk.page_content.lower()
        source = chunk.metadata.get("source", "").lower()
        return (
            sum(1 for pattern in term_patterns if pattern.search(text)),
            sum(1 for pattern in term_patterns if pattern.search(source)),
            chunk.metadata.get("source", ""),
        )

    return sorted(chunks, key=score, reverse=True)


def fallback_web_search(query: str) -> list[Result]:
    """Search the web when local context is missing."""
    html_results = _fallback_web_search_html(query)
    if html_results:
        return html_results

    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = ddgs.text(query, region="us-en", max_results=WEB_SEARCH_K)
        return [
            Result(
                page_content=r.get("body", ""),
                metadata={"source": r.get("href", "Web Search"), "doc_type": "web"}
            )
            for r in (results or [])
        ]
    except Exception as e:
        print(f"Web search failed: {e}")
        return []


def _fallback_web_search_html(query: str) -> list[Result]:
    """Fallback HTML scraping path when ddgs cannot complete HTTPS requests."""
    try:
        import certifi
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            },
            timeout=15,
            verify=certifi.where(),
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for node in soup.select(".result")[:WEB_SEARCH_K]:
            link = node.select_one(".result__a")
            snippet = node.select_one(".result__snippet")
            if not link:
                continue
            href = _unwrap_search_result_url(link.get("href", "").strip())
            body = snippet.get_text(" ", strip=True) if snippet else ""
            if href:
                results.append(
                    Result(
                        page_content=body,
                        metadata={"source": href, "doc_type": "web"},
                    )
                )
        return results
    except Exception as e:
        print(f"HTML web search fallback failed: {e}")
        return []


def _unwrap_search_result_url(url: str) -> str:
    """Extract the real destination from DuckDuckGo redirect URLs."""
    if not url:
        return ""

    if url.startswith("//"):
        url = f"https:{url}"
    elif url.startswith("/"):
        url = f"https://duckduckgo.com{url}"

    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com"):
        params = parse_qs(parsed.query)
        target = params.get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


def fetch_context(question: str, session_db_path: str | None = None, provider: str = "Local (LM Studio)") -> list[Result]:
    """Full RAG retrieval: semantic search only, then rerank."""
    rewritten = rewrite_query(question, provider)

    chunks1 = fetch_context_unranked(question, session_db_path)
    chunks2 = fetch_context_unranked(rewritten, session_db_path)
    merged = merge_chunks(chunks1, chunks2)

    if not merged:
        return []

    reranked = rerank(question, merged)[:FINAL_K]

    # Post-rerank relevance gate
    # Prevents topically-irrelevant chunks from reaching the LLM.
    if reranked:
        combined_query = f"{question} {rewritten}"
        terms = {t for t in re.findall(r"[A-Za-z0-9]+", combined_query.lower()) if len(t) > 1 and t not in _STOP_WORDS}
        if not terms:
            return []
        term_patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]
        has_overlap = any(
            any(pattern.search(c.page_content.lower()) for pattern in term_patterns)
            for c in reranked
        )
        if not has_overlap:
            return []

    return reranked


def _filter_relevant_chunks(question: str, rewritten: str, chunks: list[Result]) -> list[Result]:
    """Keep only reranked chunks that still overlap meaningfully with the query."""
    reranked = rerank(question, chunks)[:FINAL_K]
    if not reranked:
        return []

    combined_query = f"{question} {rewritten}"
    terms = {t for t in re.findall(r"[A-Za-z0-9]+", combined_query.lower()) if len(t) > 1 and t not in _STOP_WORDS}
    if not terms:
        return []

    term_patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]
    has_overlap = any(
        any(pattern.search(c.page_content.lower()) for pattern in term_patterns)
        for c in reranked
    )
    return reranked if has_overlap else []


def fetch_context_with_options(
    question: str,
    session_db_path: str | None = None,
    provider: str = "Local (LM Studio)",
    allow_web_fallback: bool = True,
    is_custom_mode: bool = False,
) -> list[Result]:
    """Retrieve context with optional suppression of web fallback for custom sessions."""
    # Custom mode is strict: ONLY search the session DB
    if is_custom_mode:
        rewritten = rewrite_query(question, provider)
        chunks1 = fetch_context_unranked(question, session_db_path)
        chunks2 = fetch_context_unranked(rewritten, session_db_path)
        merged = merge_chunks(chunks1, chunks2)
        
        if not merged:
            if allow_web_fallback:
                print("No session context found. Falling back to web search...")
                return fallback_web_search(question)
            return []
            
        filtered = _filter_relevant_chunks(question, rewritten, merged)
        if not filtered:
            if allow_web_fallback:
                print("Session context irrelevant. Falling back to web search...")
                return fallback_web_search(question)
        return filtered

    # Default mode: Search both global KB and session (session_db_path)
    return fetch_context(question, session_db_path=session_db_path, provider=provider)


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
    session_id: str | None = None,
    provider: str = "Local (LM Studio)",
    allow_web_fallback: bool = True,
    is_custom_mode: bool = False,
) -> tuple[str, list]:
    """Answer a question using the RAG pipeline. Returns (answer, context_docs)."""
    from core.session_manager import get_session_db_path
    from core.session_ingest import ingest_document

    if history is None:
        history = []

    session_db_path = get_session_db_path(session_id) if session_id else None

    # Resolve follow-up questions into standalone queries for retrieval.
    # The original question is still used for the LLM answer to preserve tone.
    retrieval_query = condense_question(question, history, provider)

    chunks = fetch_context_with_options(
        retrieval_query,
        session_db_path=session_db_path,
        provider=provider,
        allow_web_fallback=allow_web_fallback,
        is_custom_mode=is_custom_mode
    )

    # If nothing was found anywhere, return immediately without calling the LLM
    # to prevent hallucination from training data on empty context.
    if not chunks:
        if session_db_path:
            if allow_web_fallback:
                msg = (
                    "This topic was not found in your uploaded documents, "
                    "and the web search did not return any reliable results either. "
                    "Try uploading a more relevant document or rephrasing your question."
                )
            else:
                msg = (
                    "This question doesn't appear to be covered by your uploaded documents. "
                    "Try asking something specific about the content you uploaded, "
                    "or upload a more relevant document."
                )
        else:
            # Default mode — web search was also tried and failed
            msg = (
                "This topic was not found in the provided documents, "
                "and the web search did not return any results either. "
                "Try uploading a relevant document or rephrasing your question."
            )
        return msg, []

    context = build_context(chunks)

    # Use the web fallback prompt when ALL results came from a web search,
    # so the LLM clearly discloses the answer is not from the user's documents.
    is_web_fallback = all(c.metadata.get("doc_type") == "web" for c in chunks)

    # Automatically store web search results in the session database for future recall
    if is_web_fallback and session_id:
        for chunk in chunks:
            # Note: We use original question as a pseudo-source for these chunks
            # so the user knows where they came from in the docs list.
            source = chunk.metadata.get("source", "Web Search")
            ingest_document(chunk.page_content, source, session_id)

    is_web_fallback = all(c.metadata.get("doc_type") == "web" for c in chunks)
    system_prompt = (
        WEB_FALLBACK_PROMPT if is_web_fallback else SYSTEM_PROMPT
    ).format(context=context)

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
