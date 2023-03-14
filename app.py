#!/usr/bin/env python

from __future__ import annotations

import os

import gradio as gr

from model import Model

DESCRIPTION = '# [UniDiffuser](https://github.com/thu-ml/unidiffuser)'

SPACE_ID = os.getenv('SPACE_ID')
if SPACE_ID is not None:
    DESCRIPTION += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'

model = Model()


def create_demo(mode_name: str) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                mode = gr.Dropdown(label='Mode',
                                   choices=[
                                       't2i',
                                       'i2t',
                                       'joint',
                                       'i',
                                       't',
                                       'i2ti2',
                                       't2i2t',
                                   ],
                                   value=mode_name,
                                   visible=False)
                prompt = gr.Text(label='Prompt',
                                 max_lines=1,
                                 visible=mode_name in ['t2i', 't2i2t'])
                image = gr.Image(label='Input image',
                                 type='filepath',
                                 visible=mode_name in ['i2t', 'i2t2i'])
                run_button = gr.Button('Run')
                with gr.Accordion('Advanced options', open=False):
                    seed = gr.Slider(
                        label='Seed',
                        minimum=-1,
                        maximum=1000000,
                        step=1,
                        value=-1,
                        info=
                        'If set to -1, a different seed will be used each time.'
                    )
                    num_steps = gr.Slider(label='Steps',
                                          minimum=1,
                                          maximum=100,
                                          value=50,
                                          step=1)
                    guidance_scale = gr.Slider(label='Guidance Scale',
                                               minimum=0.1,
                                               maximum=30.0,
                                               value=7.0,
                                               step=0.1)
            with gr.Column():
                result_image = gr.Image(label='Generated image',
                                        visible=mode_name
                                        in ['t2i', 'i', 'joint', 'i2t2i'])
                result_text = gr.Text(label='Generated text',
                                      visible=mode_name
                                      in ['i2t', 't', 'joint', 't2i2t'])
        inputs = [
            mode,
            prompt,
            image,
            seed,
            num_steps,
            guidance_scale,
        ]
        outputs = [
            result_image,
            result_text,
        ]

        prompt.submit(fn=model.run, inputs=inputs, outputs=outputs)
        run_button.click(fn=model.run, inputs=inputs, outputs=outputs)
    return demo


with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('text2image'):
            create_demo('t2i')
        with gr.TabItem('image2text'):
            create_demo('i2t')
        with gr.TabItem('image variation'):
            create_demo('i2t2i')
        with gr.TabItem('joint generation'):
            create_demo('joint')
        with gr.TabItem('image generation'):
            create_demo('i')
        with gr.TabItem('text generation'):
            create_demo('t')
        with gr.TabItem('text variation'):
            create_demo('t2i2t')
demo.queue(api_open=False).launch()
