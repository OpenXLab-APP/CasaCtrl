import os
import gradio as gr
import numpy as np
import torch
from diffusers import DDIMScheduler
from pytorch_lightning import seed_everything

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import (AttentionBase,
                                     regiter_attention_editor_diffusers)

torch.set_grad_enabled(False)

from gradio_app.image_synthesis_app import create_demo_synthesis
from gradio_app.real_image_editing_app import create_demo_editing

from gradio_app.app_utils import global_context

SPACE_ID = os.getenv('SPACE_ID')
TITLE = '# [MasaCtrl](https://ljzycmd.github.io/projects/MasaCtrl/)</h1>'
DESCRIPTION = '<div align="center">'
DESCRIPTION += f'<p>Gradio demo for MasaCtrl: <a href="https://github.com/TencentARC/MasaCtrl">[Github]</a>, <a href="https://arxiv.org/abs/2304.08465">[Paper]</a>. If MasaCtrl is helpful, please help to ⭐ the <a href="https://github.com/TencentARC/MasaCtrl">Github Repo</a> 😊</p>'
DESCRIPTION += f'<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'
DESCRIPTION += '</div>'

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(TITLE)
    gr.HTML(DESCRIPTION)
    model_path_gr = gr.Dropdown(
        ["xyn-ai/anything-v4.0",
         "CompVis/stable-diffusion-v1-4",
         "runwayml/stable-diffusion-v1-5"],
        value="xyn-ai/anything-v4.0",
        label="Model", info="Select the model to use!"
    )
    with gr.Tab("Consistent Synthesis"):
        create_demo_synthesis()
    with gr.Tab("Real Editing"):
        create_demo_editing()

    def reload_ckpt(model_path):
        print("Reloading model from", model_path)
        global_context["model"] = MasaCtrlPipeline.from_pretrained(
            model_path, scheduler=global_context["scheduler"]).to(global_context["device"])

    model_path_gr.select(
        reload_ckpt,
        [model_path_gr]
    )


if __name__ == "__main__":
    demo.launch()
