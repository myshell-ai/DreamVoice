import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, sr=24000, frame_length=1920, hop_length=480, n_mel=128, f_min=0, f_max=12000,):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.mel = transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=frame_length,
            win_length=frame_length,
            hop_length=hop_length,
            center=False,
            power=1.0,
            norm="slaney",
            n_mels=n_mel,
            mel_scale="slaney",
            f_min=f_min,
            f_max=f_max
        )

    @torch.no_grad()
    def forward(self, x, target_length=None):
        x = F.pad(x, ((self.frame_length - self.hop_length) // 2,
                      (self.frame_length - self.hop_length) // 2), "reflect")
        mel = self.mel(x)

        target_length = mel.shape[-1] if target_length is None else target_length
        logmel = torch.zeros(mel.shape[0], mel.shape[1], target_length).to(mel.device)
        logmel[:, :, :mel.shape[2]] = mel

        logmel = torch.log(torch.clamp(logmel, min=1e-5))
        return logmel