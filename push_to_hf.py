#!/usr/bin/env python3
"""Upload hf_repo/ to Hugging Face Hub (Emma5099/dance2music)."""

import os
import sys

from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.errors import LocalTokenNotFoundError

REPO_ID = "Emma5099/dance2music"
FOLDER = os.path.join(os.path.dirname(__file__), "hf_repo")


def main() -> None:
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
            "Run:  hf auth login\n"
            "Or:   HF_TOKEN=hf_... python push_to_hf.py",
            file=sys.stderr,
        )
        sys.exit(1)

    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=FOLDER,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Release epoch 180 checkpoint with inference code",
        ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"],
    )
    print(f"Uploaded → https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
