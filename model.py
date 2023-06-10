from __future__ import annotations

import PIL.Image
import torch
from diffusers import UniDiffuserPipeline


class Model:
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            self.pipe = UniDiffuserPipeline.from_pretrained(
                'thu-ml/unidiffuser-v1', torch_dtype=torch.float16)
            self.pipe.to(self.device)
        else:
            self.pipe = UniDiffuserPipeline.from_pretrained(
                'thu-ml/unidiffuser-v1')

    def run(
        self,
        mode: str,
        prompt: str,
        image: PIL.Image.Image | None,
        seed: int = 0,
        num_steps: int = 20,
        guidance_scale: float = 8.0,
    ) -> tuple[PIL.Image.Image | None, str]:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        if mode == 't2i':
            self.pipe.set_text_to_image_mode()
            sample = self.pipe(prompt=prompt,
                               num_inference_steps=num_steps,
                               guidance_scale=guidance_scale,
                               generator=generator)
            return sample.images[0], ''
        elif mode == 'i2t':
            self.pipe.set_image_to_text_mode()
            sample = self.pipe(image=image,
                               num_inference_steps=num_steps,
                               guidance_scale=guidance_scale,
                               generator=generator)
            return None, sample.text[0]
        elif mode == 'joint':
            self.pipe.set_joint_mode()
            sample = self.pipe(num_inference_steps=num_steps,
                               guidance_scale=guidance_scale,
                               generator=generator)
            return sample.images[0], sample.text[0]
        elif mode == 'i':
            self.pipe.set_image_mode()
            sample = self.pipe(num_inference_steps=num_steps,
                               guidance_scale=guidance_scale,
                               generator=generator)
            return sample.images[0], ''
        elif mode == 't':
            self.pipe.set_text_mode()
            sample = self.pipe(num_inference_steps=num_steps,
                               guidance_scale=guidance_scale,
                               generator=generator)
            return None, sample.text[0]
        elif mode == 'i2t2i':
            self.pipe.set_image_to_text_mode()
            sample = self.pipe(image=image,
                               num_inference_steps=num_steps,
                               guidance_scale=guidance_scale,
                               generator=generator)
            self.pipe.set_text_to_image_mode()
            sample = self.pipe(prompt=sample.text[0],
                               num_inference_steps=num_steps,
                               guidance_scale=guidance_scale,
                               generator=generator)
            return sample.images[0], ''
        elif mode == 't2i2t':
            self.pipe.set_text_to_image_mode()
            sample = self.pipe(prompt=prompt,
                               num_inference_steps=num_steps,
                               guidance_scale=guidance_scale,
                               generator=generator)
            self.pipe.set_image_to_text_mode()
            sample = self.pipe(image=sample.images[0],
                               num_inference_steps=num_steps,
                               guidance_scale=guidance_scale,
                               generator=generator)
            return None, sample.text[0]
        else:
            raise ValueError
