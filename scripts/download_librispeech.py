from pathlib import Path

import torchaudio

parts = ["train-clean-100", "test-clean"]


def main():
    root = Path("data/librispeech")
    root.mkdir(parents=True, exist_ok=True)

    for part in parts:
        print("download", part)
        torchaudio.datasets.LIBRISPEECH(str(root), url=part, download=True)


if __name__ == "__main__":
    main()
