import os
import torch
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
# from feats.hubert_model import get_soft_model, get_hubert_soft_content
from feats.contentvec_hf import get_content_model, get_content
# from modules.speaker_encoder.encoder import inference as spk_encoder
# from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process
import pyworld as pw


def resample_save(infolder, audio_path, model,
                  audio_sr=24000, content_sr=16000, min_length=1.92,
                  content_resolution=50,
                  save_path='features'):
    if os.path.isfile(save_path + '/' + 'audio_24k/' + audio_path) is False:
        audio, sr = librosa.load(infolder + audio_path, sr=content_sr)
        final_length = audio.shape[-1] // (content_sr / content_resolution) * (content_sr / content_resolution)
        # final_length = final_length / content_sr

        length = max(round(min_length*content_sr), round(final_length))
        assert length % 10 == 0
        audio = audio[:length]
        audio_save = np.zeros(length, dtype=audio.dtype)
        audio_save[:audio.shape[-1]] = audio[:audio.shape[-1]]

        # content = get_hubert_soft_content(model, torch.tensor(audio_save).unsqueeze(0))
        content = get_content(model, torch.tensor(audio_save).unsqueeze(0))
        content = content.cpu()
        os.makedirs(os.path.dirname(save_path + '/' + 'content/' + audio_path), exist_ok=True)
        torch.save(content, save_path + '/' + 'content/' + audio_path+'.pt')
        # print(audio_save.shape)
        # print(content.shape)
        os.makedirs(os.path.dirname(save_path + '/' + 'audio_16k/' + audio_path), exist_ok=True)
        sf.write(save_path + '/' + 'audio_16k/' + audio_path, audio_save, int(sr))
        # print(save_path + '/' + 'audio_16k/' + audio_path)

        audio, sr = librosa.load(infolder + audio_path, sr=audio_sr)
        length = max(round(min_length*audio_sr), round(final_length/content_sr*audio_sr))
        assert length % 10 == 0
        audio = audio[:length]
        audio_save = np.zeros(length, dtype=audio.dtype)
        audio_save[:audio.shape[-1]] = audio[:audio.shape[-1]]
        # print(audio_save.shape)
        os.makedirs(os.path.dirname(save_path + '/' + 'audio_24k/' + audio_path), exist_ok=True)
        sf.write(save_path + '/' + 'audio_24k/' + audio_path, audio_save, int(sr))


def extract_f0(in_folder, audio_path, save_path):
    audio, sr = librosa.load(in_folder + audio_path, sr=None)
    assert sr == 16000
    if os.path.isfile(save_path + '/' + 'f0/' + audio_path + '.pt') is False:
        # wav = audio
        # wav = np.pad(wav, int((1024-320)/2), mode='reflect')
        # f0_, _, _ = librosa.pyin(wav, frame_length=1024, hop_length=320, center=False, sr=sr,
        #                         fmin=librosa.note_to_hz('C2'),
        #                         fmax=librosa.note_to_hz('C6'))

        _f0, t = pw.dio(audio.astype(np.float64), sr, frame_period=320 / sr * 1000)
        f0 = pw.stonemask(audio.astype(np.float64), _f0, t, sr)[:-1]

        f0 = np.nan_to_num(f0)
        os.makedirs(os.path.dirname(save_path + '/' + 'f0/' + audio_path), exist_ok=True)
        # print(save_path + '/' + 'f0/' + audio_path + '.pt')
        torch.save(torch.tensor(f0), save_path + '/' + 'f0/' + audio_path + '.pt')


def chunks(arr, m):
    result = [[] for i in range(m)]
    for i in range(len(arr)):
        result[i%m].append(arr[i])
    return result


def extract_f0_main(in_folder, audio_paths, save_path):
    for audio_path in tqdm(audio_paths):
        extract_f0(in_folder, audio_path, save_path)


if __name__ == '__main__':
    df = pd.read_csv('../test_data/vc_meta.csv')
    # model = get_soft_model('../pre_ckpts/hubert_soft.pt').to('cuda')
    model = get_content_model().to('cuda')
    # # spk_encoder.load_model(Path('ckpts/spk_encoder/pretrained.pt'), device="cuda")
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        in_path = row['path']
        resample_save('../test_data/', in_path, model, save_path='../features/')

    in_folder = '../features/audio_16k/'
    audio_files = list(df['path'])
    save_path = '../features/'
    cores = 6

    subsets = chunks(audio_files, cores)

    for subset in subsets:
        t = Process(target=extract_f0_main, args=(in_folder, subset, save_path))
        t.start()