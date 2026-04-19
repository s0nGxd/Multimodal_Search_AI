"""
Persistence Service — Syncs data to/from a HF Dataset repo.

On startup:  downloads LanceDB index + images from the repo.
After upload: pushes updated data back (in a background thread).

This solves the "Amnesia Problem" on HF Spaces free tier,
where container restarts wipe all runtime data.
"""

import os
import threading
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

# Config from environment
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
DATA_DIR = os.getenv("DATA_DIR", "data")
LANCEDB_URI = os.getenv("LANCEDB_URI", "./lancedb_db")

# Lock to prevent concurrent syncs
_sync_lock = threading.Lock()


def is_configured() -> bool:
    """Check if persistence is configured (repo + token are set)."""
    return bool(HF_DATASET_REPO and HF_TOKEN)


def restore_from_repo():
    """Download saved data from HF Dataset repo on startup.
    
    Pulls both the LanceDB directory and the images directory
    into their expected local paths. Runs synchronously at startup
    so data is available before the first search request.
    """
    if not is_configured():
        print("[Persistence] Not configured (HF_DATASET_REPO or HF_TOKEN missing). Skipping restore.")
        return

    print(f"[Persistence] Restoring data from repo: {HF_DATASET_REPO}")
    try:
        # snapshot_download pulls the entire repo into a local cache
        # and returns the path to the cached directory
        local_snapshot = snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="/tmp/hf_persistence_cache",
        )

        snapshot_path = Path(local_snapshot)

        # Restore LanceDB directory
        cached_lancedb = snapshot_path / "lancedb_db"
        if cached_lancedb.exists():
            target_lancedb = Path(LANCEDB_URI)
            target_lancedb.mkdir(parents=True, exist_ok=True)
            _copy_tree(cached_lancedb, target_lancedb)
            print(f"[Persistence] Restored LanceDB from repo → {target_lancedb}")
        else:
            print("[Persistence] No LanceDB data found in repo (first run?).")

        # Restore images directory
        cached_images = snapshot_path / "images"
        if cached_images.exists():
            target_images = Path(DATA_DIR)
            target_images.mkdir(parents=True, exist_ok=True)
            _copy_tree(cached_images, target_images)
            print(f"[Persistence] Restored images from repo → {target_images}")
        else:
            print("[Persistence] No images found in repo (first run?).")

        print("[Persistence] Restore complete.")

    except Exception as e:
        # Don't crash the app if restore fails — just start fresh
        print(f"[Persistence] Restore failed (starting fresh): {e}")


def sync_to_repo():
    """Push current data to HF Dataset repo.
    
    Called after each successful ingestion. Runs in a background
    thread so it doesn't block the API response.
    """
    if not is_configured():
        return

    def _sync():
        if not _sync_lock.acquire(blocking=False):
            print("[Persistence] Sync already in progress, skipping.")
            return
        try:
            api = HfApi(token=HF_TOKEN)

            lancedb_path = Path(LANCEDB_URI)
            images_path = Path(DATA_DIR)

            # Upload LanceDB directory
            if lancedb_path.exists() and any(lancedb_path.iterdir()):
                api.upload_folder(
                    repo_id=HF_DATASET_REPO,
                    repo_type="dataset",
                    folder_path=str(lancedb_path),
                    path_in_repo="lancedb_db",
                    commit_message="Sync LanceDB index",
                )
                print("[Persistence] Synced LanceDB to repo.")

            # Upload images directory
            if images_path.exists() and any(images_path.iterdir()):
                api.upload_folder(
                    repo_id=HF_DATASET_REPO,
                    repo_type="dataset",
                    folder_path=str(images_path),
                    path_in_repo="images",
                    commit_message="Sync uploaded images",
                )
                print("[Persistence] Synced images to repo.")

            print("[Persistence] Sync complete.")
        except Exception as e:
            print(f"[Persistence] Sync failed: {e}")
        finally:
            _sync_lock.release()

    thread = threading.Thread(target=_sync, daemon=True)
    thread.start()


def _copy_tree(src: Path, dst: Path):
    """Recursively copy files from src to dst, creating dirs as needed."""
    import shutil
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            target.mkdir(exist_ok=True)
            _copy_tree(item, target)
        else:
            shutil.copy2(item, target)
