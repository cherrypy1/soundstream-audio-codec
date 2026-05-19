from torch.utils.data import Dataset

from src.soundstream.dataset.audio_utils import load_audio, crop_audio


class LibriSpeechDataset(Dataset):
    def __init__(self, files, sample_rate, crop_samples=None, random_crop=True):
        self.files = list(files)
        self.sample_rate = sample_rate
        self.crop_samples = crop_samples
        self.random_crop = random_crop

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = load_audio(self.files[idx], self.sample_rate)

        if self.crop_samples is not None:
            audio = crop_audio(audio, self.crop_samples, self.random_crop)

        return {"audio": audio}



