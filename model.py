from __future__ import annotations

import pathlib
import random
import sys
from typing import Callable

import clip
import einops
import numpy as np
import PIL.Image
import torch
from huggingface_hub import snapshot_download

repo_dir = pathlib.Path(__file__).parent
submodule_dir = repo_dir / 'unidiffuser'
sys.path.append(submodule_dir.as_posix())

import utils
from configs.sample_unidiffuser_v1 import get_config
from dpm_solver_pp import DPM_Solver, NoiseScheduleVP
from libs.autoencoder import FrozenAutoencoderKL
from libs.autoencoder import get_model as get_autoencoder
from libs.caption_decoder import CaptionDecoder
from libs.clip import FrozenCLIPEmbedder

model_dir = repo_dir / 'models'
if not model_dir.exists():
    snapshot_download('thu-ml/unidiffuser-v1',
                      repo_type='model',
                      local_dir=model_dir)


def stable_diffusion_beta_schedule(linear_start=0.00085,
                                   linear_end=0.0120,
                                   n_timestep=1000):
    _betas = (torch.linspace(linear_start**0.5,
                             linear_end**0.5,
                             n_timestep,
                             dtype=torch.float64)**2)
    return _betas.numpy()


class Model:
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = get_config()

        self.nnet = self.load_model()
        self.caption_decoder = CaptionDecoder(device=self.device,
                                              **self.config.caption_decoder)
        self.clip_text_model = self.load_clip_text_model()
        self.autoencoder = self.load_autoencoder()

        self.clip_img_model, self.clip_img_model_preprocess = clip.load(
            'ViT-B/32', device=self.device, jit=False)
        self.empty_context = self.clip_text_model.encode([''])[0]

        self.betas = stable_diffusion_beta_schedule()
        self.N = len(self.betas)

    @property
    def use_caption_decoder(self) -> bool:
        return (self.config.text_dim < self.config.clip_text_dim
                or self.config.mode != 't2i')

    def load_model(self,
                   model_path: str = 'models/uvit_v1.pth') -> torch.nn.Module:
        model = utils.get_nnet(**self.config.nnet)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(self.device)
        model.eval()
        return model

    def load_clip_text_model(self) -> FrozenCLIPEmbedder:
        clip_text_model = FrozenCLIPEmbedder(device=self.device)
        clip_text_model.to(self.device)
        clip_text_model.eval()
        return clip_text_model

    def load_autoencoder(self) -> FrozenAutoencoderKL:
        autoencoder = get_autoencoder(**self.config.autoencoder)
        autoencoder.to(self.device)
        return autoencoder

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        C, H, W = self.config.z_shape
        z_dim = C * H * W
        z, clip_img = x.split([z_dim, self.config.clip_img_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img,
                                    'B (L D) -> B L D',
                                    L=1,
                                    D=self.config.clip_img_dim)
        return z, clip_img

    @staticmethod
    def combine(z, clip_img):
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        return torch.concat([z, clip_img], dim=-1)

    def t2i_nnet(
            self, x, timesteps, text
    ):  # text is the low dimension version of the text clip embedding
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
            config.sample.t2i_cfg_mode == 'empty_token': using the original cfg with the empty string
            config.sample.t2i_cfg_mode == 'true_uncond: using the unconditional model learned by our method
        3. return linear combination of conditional output and unconditional output
        """
        z, clip_img = self.split(x)

        t_text = torch.zeros(timesteps.size(0),
                             dtype=torch.int,
                             device=self.device)

        z_out, clip_img_out, text_out = self.nnet(
            z,
            clip_img,
            text=text,
            t_img=timesteps,
            t_text=t_text,
            data_type=torch.zeros_like(
                t_text, device=self.device, dtype=torch.int) +
            self.config.data_type)
        x_out = self.combine(z_out, clip_img_out)

        if self.config.sample.scale == 0.:
            return x_out

        if self.config.sample.t2i_cfg_mode == 'empty_token':
            _empty_context = einops.repeat(self.empty_context,
                                           'L D -> B L D',
                                           B=x.size(0))
            if self.use_caption_decoder:
                _empty_context = self.caption_decoder.encode_prefix(
                    _empty_context)
            z_out_uncond, clip_img_out_uncond, text_out_uncond = self.nnet(
                z,
                clip_img,
                text=_empty_context,
                t_img=timesteps,
                t_text=t_text,
                data_type=torch.zeros_like(
                    t_text, device=self.device, dtype=torch.int) +
                self.config.data_type)
            x_out_uncond = self.combine(z_out_uncond, clip_img_out_uncond)
        elif self.config.sample.t2i_cfg_mode == 'true_uncond':
            text_N = torch.randn_like(text)  # 3 other possible choices
            z_out_uncond, clip_img_out_uncond, text_out_uncond = self.nnet(
                z,
                clip_img,
                text=text_N,
                t_img=timesteps,
                t_text=torch.ones_like(timesteps) * self.N,
                data_type=torch.zeros_like(
                    t_text, device=self.device, dtype=torch.int) +
                self.config.data_type)
            x_out_uncond = self.combine(z_out_uncond, clip_img_out_uncond)
        else:
            raise NotImplementedError

        return x_out + self.config.sample.scale * (x_out - x_out_uncond)

    def i_nnet(self, x, timesteps):
        z, clip_img = self.split(x)
        text = torch.randn(x.size(0),
                           77,
                           self.config.text_dim,
                           device=self.device)
        t_text = torch.ones_like(timesteps) * self.N
        z_out, clip_img_out, text_out = self.nnet(
            z,
            clip_img,
            text=text,
            t_img=timesteps,
            t_text=t_text,
            data_type=torch.zeros_like(
                t_text, device=self.device, dtype=torch.int) +
            self.config.data_type)
        x_out = self.combine(z_out, clip_img_out)
        return x_out

    def t_nnet(self, x, timesteps):
        z = torch.randn(x.size(0), *self.config.z_shape, device=self.device)
        clip_img = torch.randn(x.size(0),
                               1,
                               self.config.clip_img_dim,
                               device=self.device)
        z_out, clip_img_out, text_out = self.nnet(
            z,
            clip_img,
            text=x,
            t_img=torch.ones_like(timesteps) * self.N,
            t_text=timesteps,
            data_type=torch.zeros_like(
                timesteps, device=self.device, dtype=torch.int) +
            self.config.data_type)
        return text_out

    def i2t_nnet(self, x, timesteps, z, clip_img):
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
        3. return linear combination of conditional output and unconditional output
        """
        t_img = torch.zeros(timesteps.size(0),
                            dtype=torch.int,
                            device=self.device)

        z_out, clip_img_out, text_out = self.nnet(
            z,
            clip_img,
            text=x,
            t_img=t_img,
            t_text=timesteps,
            data_type=torch.zeros_like(
                t_img, device=self.device, dtype=torch.int) +
            self.config.data_type)

        if self.config.sample.scale == 0.:
            return text_out

        z_N = torch.randn_like(z)  # 3 other possible choices
        clip_img_N = torch.randn_like(clip_img)
        z_out_uncond, clip_img_out_uncond, text_out_uncond = self.nnet(
            z_N,
            clip_img_N,
            text=x,
            t_img=torch.ones_like(timesteps) * self.N,
            t_text=timesteps,
            data_type=torch.zeros_like(
                timesteps, device=self.device, dtype=torch.int) +
            self.config.data_type)

        return text_out + self.config.sample.scale * (text_out -
                                                      text_out_uncond)

    def split_joint(self, x):
        C, H, W = self.config.z_shape
        z_dim = C * H * W
        z, clip_img, text = x.split(
            [z_dim, self.config.clip_img_dim, 77 * self.config.text_dim],
            dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img,
                                    'B (L D) -> B L D',
                                    L=1,
                                    D=self.config.clip_img_dim)
        text = einops.rearrange(text,
                                'B (L D) -> B L D',
                                L=77,
                                D=self.config.text_dim)
        return z, clip_img, text

    @staticmethod
    def combine_joint(z: torch.Tensor, clip_img: torch.Tensor,
                      text: torch.Tensor) -> torch.Tensor:
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        text = einops.rearrange(text, 'B L D -> B (L D)')
        return torch.concat([z, clip_img, text], dim=-1)

    def joint_nnet(self, x, timesteps):
        z, clip_img, text = self.split_joint(x)
        z_out, clip_img_out, text_out = self.nnet(
            z,
            clip_img,
            text=text,
            t_img=timesteps,
            t_text=timesteps,
            data_type=torch.zeros_like(
                timesteps, device=self.device, dtype=torch.int) +
            self.config.data_type)
        x_out = self.combine_joint(z_out, clip_img_out, text_out)

        if self.config.sample.scale == 0.:
            return x_out

        z_noise = torch.randn(x.size(0),
                              *self.config.z_shape,
                              device=self.device)
        clip_img_noise = torch.randn(x.size(0),
                                     1,
                                     self.config.clip_img_dim,
                                     device=self.device)
        text_noise = torch.randn(x.size(0),
                                 77,
                                 self.config.text_dim,
                                 device=self.device)

        _, _, text_out_uncond = self.nnet(
            z_noise,
            clip_img_noise,
            text=text,
            t_img=torch.ones_like(timesteps) * self.N,
            t_text=timesteps,
            data_type=torch.zeros_like(
                timesteps, device=self.device, dtype=torch.int) +
            self.config.data_type)
        z_out_uncond, clip_img_out_uncond, _ = self.nnet(
            z,
            clip_img,
            text=text_noise,
            t_img=timesteps,
            t_text=torch.ones_like(timesteps) * self.N,
            data_type=torch.zeros_like(
                timesteps, device=self.device, dtype=torch.int) +
            self.config.data_type)

        x_out_uncond = self.combine_joint(z_out_uncond, clip_img_out_uncond,
                                          text_out_uncond)

        return x_out + self.config.sample.scale * (x_out - x_out_uncond)

    @torch.cuda.amp.autocast()
    def encode(self, _batch):
        return self.autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(self, _batch):
        return self.autoencoder.decode(_batch)

    def prepare_contexts(
            self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        resolution = self.config.z_shape[-1] * 8

        contexts = torch.randn(self.config.n_samples, 77,
                               self.config.clip_text_dim).to(self.device)
        img_contexts = torch.randn(self.config.n_samples,
                                   2 * self.config.z_shape[0],
                                   self.config.z_shape[1],
                                   self.config.z_shape[2])
        clip_imgs = torch.randn(self.config.n_samples, 1,
                                self.config.clip_img_dim)

        if self.config.mode in ['t2i', 't2i2t']:
            prompts = [self.config.prompt] * self.config.n_samples
            contexts = self.clip_text_model.encode(prompts)

        elif self.config.mode in ['i2t', 'i2t2i']:
            img_contexts = []
            clip_imgs = []

            def get_img_feature(image):
                image = np.array(image).astype(np.uint8)
                image = utils.center_crop(resolution, resolution, image)
                clip_img_feature = self.clip_img_model.encode_image(
                    self.clip_img_model_preprocess(
                        PIL.Image.fromarray(image)).unsqueeze(0).to(
                            self.device))

                image = (image / 127.5 - 1.0).astype(np.float32)
                image = einops.rearrange(image, 'h w c -> 1 c h w')
                image = torch.tensor(image, device=self.device)
                moments = self.autoencoder.encode_moments(image)

                return clip_img_feature, moments

            image = PIL.Image.open(self.config.img).convert('RGB')
            clip_img, img_context = get_img_feature(image)

            img_contexts.append(img_context)
            clip_imgs.append(clip_img)
            img_contexts = img_contexts * self.config.n_samples
            clip_imgs = clip_imgs * self.config.n_samples

            img_contexts = torch.concat(img_contexts, dim=0)
            clip_imgs = torch.stack(clip_imgs, dim=0)

        return contexts, img_contexts, clip_imgs

    @staticmethod
    def unpreprocess(v: torch.Tensor) -> torch.Tensor:  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    def get_sample_fn(self, _n_samples: int) -> Callable:
        def sample_fn(mode: str, **kwargs):
            _z_init = torch.randn(_n_samples,
                                  *self.config.z_shape,
                                  device=self.device)
            _clip_img_init = torch.randn(_n_samples,
                                         1,
                                         self.config.clip_img_dim,
                                         device=self.device)
            _text_init = torch.randn(_n_samples,
                                     77,
                                     self.config.text_dim,
                                     device=self.device)
            if mode == 'joint':
                _x_init = self.combine_joint(_z_init, _clip_img_init,
                                             _text_init)
            elif mode in ['t2i', 'i']:
                _x_init = self.combine(_z_init, _clip_img_init)
            elif mode in ['i2t', 't']:
                _x_init = _text_init
            noise_schedule = NoiseScheduleVP(schedule='discrete',
                                             betas=torch.tensor(
                                                 self.betas,
                                                 device=self.device).float())

            def model_fn(x, t_continuous):
                t = t_continuous * self.N
                if mode == 'joint':
                    return self.joint_nnet(x, t)
                elif mode == 't2i':
                    return self.t2i_nnet(x, t, **kwargs)
                elif mode == 'i2t':
                    return self.i2t_nnet(x, t, **kwargs)
                elif mode == 'i':
                    return self.i_nnet(x, t)
                elif mode == 't':
                    return self.t_nnet(x, t)

            dpm_solver = DPM_Solver(model_fn,
                                    noise_schedule,
                                    predict_x0=True,
                                    thresholding=False)
            with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type):
                x = dpm_solver.sample(_x_init,
                                      steps=self.config.sample.sample_steps,
                                      eps=1. / self.N,
                                      T=1.)

            if mode == 'joint':
                _z, _clip_img, _text = self.split_joint(x)
                return _z, _clip_img, _text
            elif mode in ['t2i', 'i']:
                _z, _clip_img = self.split(x)
                return _z, _clip_img
            elif mode in ['i2t', 't']:
                return x

        return sample_fn

    @staticmethod
    def to_pil(tensor: torch.Tensor) -> PIL.Image.Image:
        return PIL.Image.fromarray(
            tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(
                'cpu', torch.uint8).numpy())

    def run(self, mode: str, prompt: str, image_path: str, seed: int,
            num_steps: int,
            guidance_scale: float) -> tuple[PIL.Image.Image | None, str]:
        self.config.mode = mode
        self.config.prompt = prompt
        self.config.img = image_path
        self.config.seed = seed
        self.config.sample.sample_steps = num_steps
        self.config.sample.scale = guidance_scale
        self.config.n_samples = 1

        #set_seed(self.config.seed)
        if seed == -1:
            seed = random.randint(0, 1000000)
        torch.manual_seed(seed)

        contexts, img_contexts, clip_imgs = self.prepare_contexts()
        if self.use_caption_decoder:
            contexts_low_dim = self.caption_decoder.encode_prefix(contexts)
        else:
            contexts_low_dim = contexts
        z_img = self.autoencoder.sample(img_contexts)

        if self.config.mode in ['t2i', 't2i2t']:
            _n_samples = contexts_low_dim.size(0)
        elif self.config.mode in ['i2t', 'i2t2i']:
            _n_samples = img_contexts.size(0)
        else:
            _n_samples = self.config.n_samples
        sample_fn = self.get_sample_fn(_n_samples)

        if self.config.mode == 'joint':
            _z, _clip_img, _text = sample_fn(self.config.mode)
            samples = self.unpreprocess(self.decode(_z))
            samples = [self.to_pil(tensor) for tensor in samples]
            prompts = self.caption_decoder.generate_captions(_text)
            return samples[0], prompts[0]

        elif self.config.mode in ['t2i', 'i', 'i2t2i']:
            if self.config.mode == 't2i':
                _z, _clip_img = sample_fn(
                    self.config.mode,
                    text=contexts_low_dim)  # conditioned on the text embedding
            elif self.config.mode == 'i':
                _z, _clip_img = sample_fn(self.config.mode)
            elif self.config.mode == 'i2t2i':
                _text = sample_fn(
                    'i2t', z=z_img,
                    clip_img=clip_imgs)  # conditioned on the image embedding
                _z, _clip_img = sample_fn('t2i', text=_text)
            samples = self.unpreprocess(self.decode(_z))
            samples = [self.to_pil(tensor) for tensor in samples]
            return samples[0], ''

        elif self.config.mode in ['i2t', 't', 't2i2t']:
            if self.config.mode == 'i2t':
                _text = sample_fn(
                    self.config.mode, z=z_img,
                    clip_img=clip_imgs)  # conditioned on the image embedding
            elif self.config.mode == 't':
                _text = sample_fn(self.config.mode)
            elif self.config.mode == 't2i2t':
                _z, _clip_img = sample_fn('t2i', text=contexts_low_dim)
                _text = sample_fn('i2t', z=_z, clip_img=_clip_img)
            prompts = self.caption_decoder.generate_captions(_text)
            return None, prompts[0]
        else:
            raise ValueError
