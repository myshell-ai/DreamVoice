import torch, torchaudio
from .hubert.hubert import HubertSoft
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import librosa


def get_soft_model(model_path):
    hubert = HubertSoft()
    # Load checkpoint (either hubert_soft or hubert_discrete)
    # hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
    checkpoint = torch.load(model_path)
    consume_prefix_in_state_dict_if_present(checkpoint["hubert"], "module.")
    hubert.load_state_dict(checkpoint["hubert"])
    hubert.eval()
    return hubert


@torch.no_grad()
def get_hubert_soft_content(hmodel, wav_16k_tensor, device='cuda'):
    wav_16k_tensor = wav_16k_tensor.to(device).unsqueeze(1)
    # print(wav_16k_tensor.shape)
    units = hmodel.units(wav_16k_tensor)
    # print(units.shape)
    return units.cpu()