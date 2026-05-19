import os
from huggingface_hub import HfApi, create_repo


def main():
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    repo_id = "artii-ml/soundstream-codec"
    create_repo(
        repo_id=repo_id,
        private=False,
        exist_ok=True,
        token=token,
    )
    api.upload_file(
        path_or_fileobj="checkpoints/final.ckpt",
        path_in_repo="final.ckpt",
        repo_id=repo_id,
    )


if __name__ == "__main__":
    main()
