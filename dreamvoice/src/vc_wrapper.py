import yaml
import torch
from diffusers import DDIMScheduler
from .model.model import DiffVC
from .model.model_cross import DiffVC_Cross
from .utils import scale_shift, scale_shift_re, rescale_noise_cfg


class ReDiffVC(object):
    def __init__(self,
                 config_path='configs/diffvc_base.yaml',
                 ckpt_path='../ckpts/dreamvc_base.pt',
                 device='cpu'):

        with open(config_path, 'r') as fp:
            config = yaml.safe_load(fp)

        self.device = device
        self.model = DiffVC(config['model']).to(device)
        self.model.load_state_dict(torch.load(ckpt_path)['model'])
        self.model.eval()

        noise_scheduler = DDIMScheduler(num_train_timesteps=config['scheduler']['num_train_steps'],
                                        beta_start=config['scheduler']['beta_start'],
                                        beta_end=config['scheduler']['beta_end'],
                                        rescale_betas_zero_snr=True,
                                        timestep_spacing="trailing",
                                        clip_sample=False,
                                        prediction_type='v_prediction')
        self.noise_scheduler = noise_scheduler
        self.scale = config['scheduler']['scale']
        self.shift = config['scheduler']['shift']
        self.melshape = config['model']['unet']['sample_size'][0]

    @torch.no_grad()
    def inference(self,
                  spk_embed, content_clip, f0_clip=None,
                  guidance_scale=3, guidance_rescale=0.7,
                  ddim_steps=50, eta=1, random_seed=2023):

        self.model.eval()
        if random_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(random_seed)
        else:
            generator = torch.Generator(device=self.device)
            generator.seed()

        self.noise_scheduler.set_timesteps(ddim_steps)

        # init noise
        gen_shape = (1, 1, self.melshape, content_clip.shape[-2])
        noise = torch.randn(gen_shape, generator=generator, device=self.device)
        latents = noise

        for t in self.noise_scheduler.timesteps:
            latents = self.noise_scheduler.scale_model_input(latents, t)

            if guidance_scale:
                output_text = self.model(latents, t, content_clip, spk_embed, f0_clip, train_cfg=False)
                output_uncond = self.model(latents, t, content_clip, spk_embed, f0_clip, train_cfg=True,
                                           speaker_cfg=1.0, pitch_cfg=0.0)

                output_pred = output_uncond + guidance_scale * (output_text - output_uncond)
                if guidance_rescale > 0.0:
                    output_pred = rescale_noise_cfg(output_pred, output_text,
                                                    guidance_rescale=guidance_rescale)
            else:
                output_pred = self.model(latents, t, content_clip, spk_embed, f0_clip, train_cfg=False)

            latents = self.noise_scheduler.step(model_output=output_pred, timestep=t, sample=latents,
                                                eta=eta, generator=generator).prev_sample

        pred = scale_shift_re(latents, scale=1/self.scale, shift=self.shift)
        return pred


class DreamVC(object):
    def __init__(self,
                 config_path='configs/diffvc_cross.yaml',
                 ckpt_path='../ckpts/dreamvc_cross.pt',
                 device='cpu'):

        with open(config_path, 'r') as fp:
            config = yaml.safe_load(fp)

        self.device = device
        self.model = DiffVC_Cross(config['model']).to(device)
        self.model.load_state_dict(torch.load(ckpt_path)['model'])
        self.model.eval()

        noise_scheduler = DDIMScheduler(num_train_timesteps=config['scheduler']['num_train_steps'],
                                        beta_start=config['scheduler']['beta_start'],
                                        beta_end=config['scheduler']['beta_end'],
                                        rescale_betas_zero_snr=True,
                                        timestep_spacing="trailing",
                                        clip_sample=False,
                                        prediction_type='v_prediction')
        self.noise_scheduler = noise_scheduler
        self.scale = config['scheduler']['scale']
        self.shift = config['scheduler']['shift']
        self.melshape = config['model']['unet']['sample_size'][0]

    @torch.no_grad()
    def inference(self,
                  text, content_clip, f0_clip=None,
                  guidance_scale=3, guidance_rescale=0.7,
                  ddim_steps=50, eta=1, random_seed=2023):

        text, text_mask = text
        self.model.eval()
        if random_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(random_seed)
        else:
            generator = torch.Generator(device=self.device)
            generator.seed()

        self.noise_scheduler.set_timesteps(ddim_steps)

        # init noise
        gen_shape = (1, 1, self.melshape, content_clip.shape[-2])
        noise = torch.randn(gen_shape, generator=generator, device=self.device)
        latents = noise

        for t in self.noise_scheduler.timesteps:
            latents = self.noise_scheduler.scale_model_input(latents, t)

            if guidance_scale:
                output_text = self.model(latents, t, content_clip, text, text_mask, f0_clip, train_cfg=False)
                output_uncond = self.model(latents, t, content_clip, text, text_mask, f0_clip, train_cfg=True,
                                           speaker_cfg=1.0, pitch_cfg=0.0)

                output_pred = output_uncond + guidance_scale * (output_text - output_uncond)
                if guidance_rescale > 0.0:
                    output_pred = rescale_noise_cfg(output_pred, output_text,
                                                    guidance_rescale=guidance_rescale)
            else:
                output_pred = self.model(latents, t, content_clip, text, text_mask, f0_clip, train_cfg=False)

            latents = self.noise_scheduler.step(model_output=output_pred, timestep=t, sample=latents,
                                                eta=eta, generator=generator).prev_sample

        pred = scale_shift_re(latents, scale=1/self.scale, shift=self.shift)
        return pred

