import torch
import librosa
from fairseq import checkpoint_utils
import torch.nn.functional as F


def get_model(vec_path):
    print("load model(s) from {}".format(vec_path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [vec_path],
        suffix="",
    )
    model = models[0]
    model.eval()
    return model


@torch.no_grad()
def get_content(hmodel, wav_16k_tensor, device='cuda', layer=12):
    # print(layer)
    wav_16k_tensor = wav_16k_tensor.to(device)
    # so that the output shape will be len(audio//320)
    wav_16k_tensor = F.pad(wav_16k_tensor, ((400 - 320) // 2, (400 - 320) // 2))
    feats = wav_16k_tensor
    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
    inputs = {
        "source": feats.to(wav_16k_tensor.device),
        "padding_mask": padding_mask.to(wav_16k_tensor.device),
        "output_layer": layer
    }
    logits = hmodel.extract_features(**inputs)[0]
    # feats = hmodel.final_proj(logits[0])
    return logits


if __name__ == '__main__':
    audio, sr = librosa.load('test.wav', sr=16000)
    audio = audio[:100*320]
    model = get_model('../../ckpts/checkpoint_best_legacy_500.pt')
    model = model.cuda()
    content = get_content(model, torch.tensor([audio]))
    print(content)