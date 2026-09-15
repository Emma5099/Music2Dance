#!/usr/bin/env python3
"""Upload hf_repo/ to Hugging Face Hub (Emma5099/dance2music)."""

import os
import sys

from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.errors import LocalTokenNotFoundError

REPO_ID = "Emma5099/dance2music"
ROOT = os.path.dirname(__file__)
FOLDER = os.path.join(ROOT, "hf_repo")
ENV_PATH = os.path.join(ROOT, ".env")


def load_dotenv(path: str = ENV_PATH) -> None:
    """Load KEY=VALUE pairs from .env into os.environ (no overwrite)."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def main() -> None:
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    api = HfApi()
    try:
        user = api.whoami()["name"]
        print(f"Authenticated as {user}")
    except LocalTokenNotFoundError:
        print(
            "Not logged in to Hugging Face.\n"
            "Put HF_TOKEN in .env, or run: hf auth login",
            file=sys.stderr,
        )
        sys.exit(1)

    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=FOLDER,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Release epoch 250 checkpoint with inference code",
        ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"],
    )
    print(f"Uploaded → https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
