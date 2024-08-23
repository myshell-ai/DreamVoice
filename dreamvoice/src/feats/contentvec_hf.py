from transformers import HubertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import librosa


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def get_content_model(config='lengyue233/content-vec-best'):
    model = HubertModelWithFinalProj.from_pretrained(config)
    model.eval()
    return model


@torch.no_grad()
def get_content(model, wav_16k_tensor, device='cuda'):
    # print(layer)
    wav_16k_tensor = wav_16k_tensor.to(device)
    # so that the output shape will be len(audio//320)
    wav_16k_tensor = F.pad(wav_16k_tensor, ((400 - 320) // 2, (400 - 320) // 2))
    logits = model(wav_16k_tensor)['last_hidden_state']
    return logits


if __name__ == '__main__':
    model = get_content_model().cuda()
    audio, sr = librosa.load('test.wav', sr=16000)
    audio = audio[:100*320]
    audio = torch.tensor([audio])
    content = get_content(model, audio, 'cuda')
    print(content)