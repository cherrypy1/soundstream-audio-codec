import torch

from src.soundstream.dataset.audio_utils import load_audio, save_audio
from src.soundstream.utils.utils import get_model


def resynthesize_sound(checkpoint_path, input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model, _ = get_model(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    audio = load_audio(input_path, config["sample_rate"]).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed = model(audio, update_codebook=False)["audio"][0]

    save_audio(output_path, reconstructed.cpu(), config["sample_rate"])
