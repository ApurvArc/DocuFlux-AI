import shutil
import tempfile
import uuid
from pathlib import Path

_ROOT = Path(__file__).parent.parent
TEMP_DIR = _ROOT / "data" / "sessions"

_session_sizes: dict[str, int] = {}
_session_processed_files: dict[str, set[str]] = {}
_session_clients = {}

def get_session_client(session_id: str):
    """Lazy-get or create exactly one Chroma PersistentClient per session."""
    if session_id not in _session_clients:
        from chromadb import PersistentClient
        db_path = get_session_db_path(session_id)
        _session_clients[session_id] = PersistentClient(path=db_path)
    return _session_clients[session_id]


def create_session() -> str:
    """Generate UUID, create temp dirs, return session_id."""
    session_id = str(uuid.uuid4())
    (TEMP_DIR / session_id / "vector_db").mkdir(parents=True, exist_ok=True)
    _session_sizes[session_id] = 0
    _session_processed_files[session_id] = set()
    return session_id


def destroy_session(session_id: str) -> None:
    """Wipe the session directory completely."""
    session_dir = TEMP_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    _session_sizes.pop(session_id, None)
    _session_processed_files.pop(session_id, None)
    _session_clients.pop(session_id, None)


def get_session_db_path(session_id: str) -> str:
    """Returns absolute path to session's vector_db dir."""
    return str(TEMP_DIR / session_id / "vector_db")


def get_session_size(session_id: str) -> int:
    """Returns current cumulative size of files added to session."""
    return _session_sizes.get(session_id, 0)


def add_to_session_size(session_id: str, size_bytes: int) -> None:
    """Increase the tracked size for the session."""
    _session_sizes[session_id] = _session_sizes.get(session_id, 0) + size_bytes


def is_file_processed(session_id: str, filename: str) -> bool:
    """Check if a file has already been ingested in this session."""
    return filename in _session_processed_files.get(session_id, set())


def mark_file_processed(session_id: str, filename: str) -> None:
    """Mark a file as ingested for this session."""
    if session_id not in _session_processed_files:
        _session_processed_files[session_id] = set()
    _session_processed_files[session_id].add(filename)


def clear_all_sessions() -> None:
    """Wipe all temp session data. Called on startup and clean exit."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    _session_sizes.clear()
    _session_processed_files.clear()
