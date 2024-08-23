import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import torch


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def scale_shift(x, scale, shift):
    return (x+shift) * scale


def scale_shift_re(x, scale, shift):
    return (x/scale) - shift


def align_seq(source, target_length, mapping_method='hard'):
    source_len = source.shape[1]
    if mapping_method == 'hard':
        mapping_idx = np.round(np.arange(target_length) * source_len / target_length)
        output = source[:, mapping_idx]
    else:
        # TBD
        raise NotImplementedError

    return output


def save_plot(tensor, savepath):
    tensor = tensor.squeeze().cpu()
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()


def save_audio(file_path, sampling_rate, audio):
    audio = np.clip(audio.cpu().squeeze().numpy(), -0.999, 0.999)
    wavfile.write(file_path, sampling_rate, (audio * 32767).astype("int16"))


def minmax_norm_diff(tensor: torch.Tensor, vmax: float = 2.5, vmin: float = -12) -> torch.Tensor:
    tensor = torch.clip(tensor, vmin, vmax)
    tensor = 2 * (tensor - vmin) / (vmax - vmin) - 1
    return tensor


def reverse_minmax_norm_diff(tensor: torch.Tensor, vmax: float = 2.5, vmin: float = -12) -> torch.Tensor:
    tensor = torch.clip(tensor, -1.0, 1.0)
    tensor = (tensor + 1) / 2
    tensor = tensor * (vmax - vmin) + vmin
    return tensor


if __name__ == "__main__":

    a = torch.rand(2, 10)
    target_len = 15

    b = align_seq(a, target_len)