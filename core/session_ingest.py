from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.ingest import get_embedder, CHUNK_SIZE, CHUNK_OVERLAP
from core.session_manager import get_session_client

def get_session_collection(session_id: str):
    """Lazy-get or create the Chroma collection for this session."""
    chroma = get_session_client(session_id)
    return chroma.get_or_create_collection("session")

def ingest_document(text: str, source_name: str, session_id: str) -> int:
    """
    Chunk text -> embed -> add to session vector_db.
    Returns number of chunks added.
    """
    if not text.strip() or text.startswith("Error:"):
        return 0
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    splits = splitter.split_text(text)
    if not splits:
        return 0
        
    chunks = []
    for split in splits:
        chunks.append({
            "text": split,
            "metadata": {
                "source": source_name,
                "doc_type": "user_upload",
                "session": session_id
            },
        })
        
    texts = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    
    collection = get_session_collection(session_id)
    
    start_id = collection.count()
    ids = [str(start_id + i) for i in range(len(chunks))]
    
    print(f"[{session_id}] Encoding {len(chunks)} chunks from {source_name}...")
    embedder = get_embedder()
    vectors = embedder.encode(texts, show_progress_bar=False, batch_size=64).tolist()
    
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    
    return len(chunks)
