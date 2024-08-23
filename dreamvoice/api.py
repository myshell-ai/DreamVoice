import os
import requests
import yaml
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
from .src.vc_wrapper import ReDiffVC, DreamVC
from .src.plugin_wrapper import DreamVG
from .src.modules.speaker_encoder.encoder import inference as spk_encoder
from .src.modules.BigVGAN.inference import load_model as load_vocoder
from .src.feats.contentvec_hf import get_content_model, get_content


class DreamVoice:
    def __init__(self, config='dreamvc.yaml', mode='plugin', device='cuda', chunk_size=16):
        # Initial setup
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / config

        # Load configuration file
        with open(config_path, 'r') as fp:
            self.config = yaml.safe_load(fp)

        self.script_dir = script_dir

        # Ensure all checkpoints are downloaded
        self._ensure_checkpoints_exist()

        # Initialize attributes
        self.device = device
        self.sr = self.config['sample_rate']

        # Load vocoder
        vocoder_path = script_dir / self.config['vocoder_path']
        self.hifigan, _ = load_vocoder(vocoder_path, device)
        self.hifigan.eval()

        # Load content model
        self.content_model = get_content_model().to(device)

        # Load tokenizer and text encoder
        lm_path = self.config['lm_path']
        self.tokenizer = T5Tokenizer.from_pretrained(lm_path)
        self.text_encoder = T5EncoderModel.from_pretrained(lm_path).to(device).eval()

        # Set mode
        self.mode = mode
        if mode == 'plugin':
            self._init_plugin_mode()
        elif mode == 'end2end':
            self._init_end2end_mode()
        else:
            raise NotImplementedError("Select mode from 'plugin' and 'end2end'")

        # chunk inputs to 10s clips
        self.chunk_size = chunk_size * 50

    def _ensure_checkpoints_exist(self):
        checkpoints = [
            ('vocoder_path', self.config.get('vocoder_url')),
            ('vocoder_config_path', self.config.get('vocoder_config_url')),
            ('speaker_path', self.config.get('speaker_url')),
            ('dreamvc.ckpt_path', self.config.get('dreamvc', {}).get('ckpt_url')),
            ('rediffvc.ckpt_path', self.config.get('rediffvc', {}).get('ckpt_url')),
            ('dreamvg.ckpt_path', self.config.get('dreamvg', {}).get('ckpt_url'))
        ]

        for path_key, url in checkpoints:
            local_path = self._get_local_path(path_key)
            if not local_path.exists() and url:
                print(f"Downloading {path_key} from {url}")
                self._download_file(url, local_path)

    def _get_local_path(self, path_key):
        keys = path_key.split('.')
        local_path = self.config
        for key in keys:
            local_path = local_path.get(key, {})
        return self.script_dir / local_path

    def _download_file(self, url, local_path):
        try:
            # Attempt to send a GET request to the URL
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Ensure we raise an exception for HTTP errors
        except requests.exceptions.RequestException as e:
            # Log the error for debugging purposes
            print(f"Error encountered: {e}")

            # Development mode: prompt user for Hugging Face API key
            user_input = input("Private checkpoint, please request authorization and enter your Hugging Face API key.")
            self.hf_key = user_input if user_input else None

            # Set headers if an API key is provided
            headers = {'Authorization': f'Bearer {self.hf_key}'} if self.hf_key else {}

            try:
                # Attempt to send a GET request with headers in development mode
                response = requests.get(url, stream=True, headers=headers)
                response.raise_for_status()  # Ensure we raise an exception for HTTP errors
            except requests.exceptions.RequestException as e:
                # Log the error for debugging purposes
                print(f"Error encountered in dev mode: {e}")
                response = None  # Handle response accordingly in your code

        local_path.parent.mkdir(parents=True, exist_ok=True)

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        t = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                t.update(len(chunk))
                f.write(chunk)
        t.close()

    def _init_plugin_mode(self):
        # Initialize ReDiffVC
        self.dreamvc = ReDiffVC(
            config_path=self.script_dir / self.config['rediffvc']['config_path'],
            ckpt_path=self.script_dir / self.config['rediffvc']['ckpt_path'],
            device=self.device
        )

        # Initialize DreamVG
        self.dreamvg = DreamVG(
            config_path=self.script_dir / self.config['dreamvg']['config_path'],
            ckpt_path=self.script_dir / self.config['dreamvg']['ckpt_path'],
            device=self.device
        )

        # Load speaker encoder
        spk_encoder.load_model(self.script_dir / self.config['speaker_path'], self.device)
        self.spk_encoder = spk_encoder
        self.spk_embed_cache = None

    def _init_end2end_mode(self):
        # Initialize DreamVC
        self.dreamvc = DreamVC(
            config_path=self.script_dir / self.config['dreamvc']['config_path'],
            ckpt_path=self.script_dir / self.config['dreamvc']['ckpt_path'],
            device=self.device
        )

    def _load_content(self, audio_path):
        content_audio, _ = librosa.load(audio_path, sr=16000)
        # Calculate the required length to make it a multiple of 16*160
        target_length = ((len(content_audio) + 16*160 - 1) // (16*160)) * (16*160)
        # Pad with zeros if necessary
        if len(content_audio) < target_length:
            content_audio = np.pad(content_audio, (0, target_length - len(content_audio)), mode='constant')
        content_audio = torch.tensor(content_audio).unsqueeze(0).to(self.device)
        content_clip = get_content(self.content_model, content_audio)
        return content_clip

    def load_spk_embed(self, emb_path):
        self.spk_embed_cache = torch.load(emb_path, map_location=self.device)

    def save_spk_embed(self, emb_path):
        assert self.spk_embed_cache is not None
        torch.save(self.spk_embed_cache.cpu(), emb_path)

    def save_audio(self, output_path, audio, sr):
        sf.write(output_path, audio, samplerate=sr)

    @torch.no_grad()
    def genvc(self, content_audio, prompt,
              prompt_guidance_scale=3, prompt_guidance_rescale=0.0,
              prompt_ddim_steps=100, prompt_eta=1, prompt_random_seed=None,
              vc_guidance_scale=3, vc_guidance_rescale=0.0,
              vc_ddim_steps=50, vc_eta=1, vc_random_seed=None,
             ):

        content_clip = self._load_content(content_audio)

        text_batch = self.tokenizer(prompt, max_length=32,
                                    padding='max_length', truncation=True, return_tensors="pt")
        text, text_mask = text_batch.input_ids.to(self.device), \
            text_batch.attention_mask.to(self.device)
        text = self.text_encoder(input_ids=text, attention_mask=text_mask)[0]

        if self.mode == 'plugin':
            spk_embed = self.dreamvg.inference([text, text_mask],
                                               guidance_scale=prompt_guidance_scale,
                                               guidance_rescale=prompt_guidance_rescale,
                                               ddim_steps=prompt_ddim_steps, eta=prompt_eta,
                                               random_seed=prompt_random_seed)
            
            B, L, D = content_clip.shape
            gen_audio_chunks = []
            num_chunks = (L + self.chunk_size - 1) // self.chunk_size
            for i in range(num_chunks):
                start_idx = i * self.chunk_size
                end_idx = min((i + 1) * self.chunk_size, L)
                content_clip_chunk = content_clip[:, start_idx:end_idx, :]
                
                gen_audio_chunk = self.dreamvc.inference(
                        spk_embed, content_clip_chunk, None,
                        guidance_scale=vc_guidance_scale,
                        guidance_rescale=vc_guidance_rescale,
                        ddim_steps=vc_ddim_steps,
                        eta=vc_eta,
                        random_seed=vc_random_seed)
                    
                gen_audio_chunks.append(gen_audio_chunk)
            
            gen_audio = torch.cat(gen_audio_chunks, dim=-1)

            self.spk_embed_cache = spk_embed

        elif self.mode == 'end2end':
            B, L, D = content_clip.shape
            gen_audio_chunks = []
            num_chunks = (L + self.chunk_size - 1) // self.chunk_size
            
            for i in range(num_chunks):
                start_idx = i * self.chunk_size
                end_idx = min((i + 1) * self.chunk_size, L)
                content_clip_chunk = content_clip[:, start_idx:end_idx, :]
                
                gen_audio_chunk = self.dreamvc.inference([text, text_mask], content_clip,
                                                         guidance_scale=prompt_guidance_scale,
                                                         guidance_rescale=prompt_guidance_rescale,
                                                         ddim_steps=prompt_ddim_steps,
                                                         eta=prompt_eta, random_seed=prompt_random_seed)
                gen_audio_chunks.append(gen_audio_chunk)
            
            gen_audio = torch.cat(gen_audio_chunks, dim=-1)

        else:
            raise NotImplementedError("Select mode from 'plugin' and 'end2end'")

        gen_audio = self.hifigan(gen_audio.squeeze(1))
        gen_audio = gen_audio.cpu().numpy().squeeze(0).squeeze(0)

        return gen_audio, self.sr

    @torch.no_grad()
    def simplevc(self, content_audio, speaker_audio=None, use_spk_cache=False,
                 vc_guidance_scale=3, vc_guidance_rescale=0.0,
                 vc_ddim_steps=50, vc_eta=1, vc_random_seed=None,
                ):

        assert self.mode == 'plugin'
        if speaker_audio is not None:
            speaker_audio, _ = librosa.load(speaker_audio, sr=16000)
            speaker_audio = torch.tensor(speaker_audio).unsqueeze(0).to(self.device)
            spk_embed = spk_encoder.embed_utterance_batch(speaker_audio)
            self.spk_embed_cache = spk_embed
        elif use_spk_cache:
            assert self.spk_embed_cache is not None
            spk_embed = self.spk_embed_cache
        else:
            raise NotImplementedError

        content_clip = self._load_content(content_audio)

        B, L, D = content_clip.shape
        gen_audio_chunks = []
        num_chunks = (L + self.chunk_size - 1) // self.chunk_size
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, L)
            content_clip_chunk = content_clip[:, start_idx:end_idx, :]
            
            gen_audio_chunk = self.dreamvc.inference(
                    spk_embed, content_clip_chunk, None,
                    guidance_scale=vc_guidance_scale,
                    guidance_rescale=vc_guidance_rescale,
                    ddim_steps=vc_ddim_steps,
                    eta=vc_eta,
                    random_seed=vc_random_seed)
                
            gen_audio_chunks.append(gen_audio_chunk)
        
        gen_audio = torch.cat(gen_audio_chunks, dim=-1)

        gen_audio = self.hifigan(gen_audio.squeeze(1))
        gen_audio = gen_audio.cpu().numpy().squeeze(0).squeeze(0)

        return gen_audio, self.sr


if __name__ == '__main__':
    dreamvoice = DreamVoice(config='dreamvc.yaml', mode='plugin', device='cuda')
    content_audio = 'test.wav'
    speaker_audio = 'speaker.wav'
    prompt = 'young female voice, sounds young and cute'
    gen_audio, sr = dreamvoice.genvc('test.wav', prompt)
    dreamvoice.save_audio('debug.wav', gen_audio, sr)