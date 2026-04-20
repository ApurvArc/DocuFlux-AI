"""Ingest the static markdown knowledge base into ChromaDB."""

import glob
from pathlib import Path

from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.answer import get_embedder as get_embedder
from core.config import DB_NAME, KNOWLEDGE_BASE, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP


def fetch_documents() -> list[dict]:
    """Load all markdown files from the knowledge base with metadata."""
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = Path(folder).name
        for filepath in glob.glob(str(Path(folder) / "**/*.md"), recursive=True):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                documents.append({
                    "text": text,
                    "source": filepath,
                    "doc_type": doc_type,
                })
    print(f"Loaded {len(documents)} documents from {len(folders)} categories")
    return documents


def create_chunks(documents: list[dict]) -> list[dict]:
    """Split documents into overlapping chunks with preserved metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for split in splits:
            chunks.append({
                "text": split,
                "metadata": {
                    "source": doc["source"],
                    "doc_type": doc["doc_type"],
                },
            })
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def create_embeddings(chunks: list[dict]) -> None:
    """Embed all chunks and store in Chroma (replaces existing collection)."""
    chroma = PersistentClient(path=DB_NAME)

    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    texts = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    ids = [str(i) for i in range(len(chunks))]

    print("Encoding chunks...")
    vectors = get_embedder().encode(texts, show_progress_bar=True, batch_size=64).tolist()

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)

    sample = collection.get(limit=1, include=["embeddings"])
    dims = len(sample["embeddings"][0])
    print(f"Vectorstore ready: {collection.count():,} vectors x {dims} dimensions")


def ingest_all(force: bool = False):
    """Run the complete ingestion pipeline if needed."""
    from chromadb import PersistentClient
    from core.config import DB_NAME, COLLECTION_NAME

    if not force:
        chroma = PersistentClient(path=DB_NAME)
        if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
            coll = chroma.get_collection(COLLECTION_NAME)
            count = coll.count()
            if count > 0:
                print(f"Vector DB already exists with {count} vectors. Skipping ingestion.")
                return

    documents = fetch_documents()
    if not documents:
        print("No documents found in knowledge base.")
        return
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")


if __name__ == "__main__":
    ingest_all()
