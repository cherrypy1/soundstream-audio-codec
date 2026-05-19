from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id="artii-ml/soundstream-codec",
        filename="final.ckpt",
        local_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main()
