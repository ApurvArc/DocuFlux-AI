import os
import subprocess
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
LOCAL_DATA_DIR = _ROOT / "data"

# Use the exact bucket URI you provided, or allow override via ENV
HF_BUCKET_URI = os.getenv("HF_BUCKET_URI", "hf://buckets/arcreactor19/DocuFlux-AI-storage")


def restore_from_bucket():
    """Download the bucket to the local data folder using the hf CLI."""
    print(f"[Sync] Downloading bucket {HF_BUCKET_URI} to {LOCAL_DATA_DIR}...")
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # hf sync hf://buckets/... ./data
        subprocess.run(["hf", "sync", HF_BUCKET_URI, str(LOCAL_DATA_DIR)], check=True, shell=os.name == 'nt')
        print("[Sync] Restore complete.")
    except subprocess.CalledProcessError as e:
        print(f"[Sync] Restore failed (CLI error): {e}")
    except FileNotFoundError:
        print("[Sync] ERROR: The 'hf' CLI is not installed or not in PATH.")


def backup_to_bucket():
    """Upload the local data folder to the bucket using the hf CLI."""
    if not LOCAL_DATA_DIR.exists():
        return
        
    print(f"[Sync] Uploading {LOCAL_DATA_DIR} to {HF_BUCKET_URI}...")
    try:
        # hf sync ./data hf://buckets/...
        subprocess.run(["hf", "sync", str(LOCAL_DATA_DIR), HF_BUCKET_URI], check=True, shell=os.name == 'nt')
        print("[Sync] Backup complete.")
    except subprocess.CalledProcessError as e:
        print(f"[Sync] Background backup failed (CLI error): {e}")
    except FileNotFoundError:
        print("[Sync] ERROR: The 'hf' CLI is not installed or not in PATH.")


def start_background_sync(interval_seconds: int = 300):
    """Start a daemon thread that periodically backs up the local databases to the HF Bucket."""
    def _sync_loop():
        print(f"[Sync] Background sync started. Backing up every {interval_seconds}s.")
        while True:
            time.sleep(interval_seconds)
            backup_to_bucket()

    thread = threading.Thread(target=_sync_loop, daemon=True)
    thread.start()
