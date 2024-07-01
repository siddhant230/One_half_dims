# !pip install -q gradio==3.50.2 rembg
import logging
import os
import tempfile
import time

import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial
import random

import argparse

rembg_session = rembg.new_session()


def background_remover_func(a, b=True, c=0.8):
    return a


def generate(a, b=300):
    return None, None


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def img_generator(img, text):
    image = img["background"]
    mask = img["layers"][0].convert('1')
    return image


def fake_gan(src_list=None, trg_list=None):
    images2 = [
        (random.choice(
              [
                  "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg",
                  "http://www.marketingtool.online/en/face-generator/img/faces/avatar-116b5e92936b766b7fdfc242649337f7.jpg",
                  "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1163530ca19b5cebe1b002b8ec67b6fc.jpg",
                  "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1116395d6e6a6581eef8b8038f4c8e55.jpg",
                  "http://www.marketingtool.online/en/face-generator/img/faces/avatar-11319be65db395d0e8e6855d18ddcef0.jpg",
              ]
        ), f"label {i}_yoooooo")
        for i in range(15)
    ]

    return images2


def display(image_list, event_data: gr.EventData):
    image_index = event_data._data["index"]
    image_name = event_data._data["value"]["image"]["path"]
    caption = event_data._data["value"]["caption"]
    return f"{image_index}-{image_name}-{caption}"


with gr.Blocks(title="1.5D") as interface:
    gr.Markdown(
        """1.5 Dimensions"""
    )
    with gr.Tab("Acquisition"):
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    input_image = gr.ImageEditor(
                        label="Input Image",
                        # tool='sketch',
                        image_mode="RGBA",
                        type="pil",
                        elem_id="content_image",
                    )

                with gr.Row():
                    with gr.Column():
                        text_prompt = gr.Text(label="PROMPT")
                    with gr.Group():
                        do_remove_background = gr.Checkbox(
                            label="Remove Background", value=True
                        )
                        foreground_ratio = gr.Slider(
                            label="Foreground Ratio",
                            minimum=0.5,
                            maximum=1.0,
                            value=0.85,
                            step=0.05,
                        )
                        mc_resolution = gr.Slider(
                            label="Marching Cubes Resolution",
                            minimum=32,
                            maximum=320,
                            value=256,
                            step=32
                        )

                with gr.Row():
                    submit = gr.Button(
                        "Generate", elem_id="generate", variant="primary")
            with gr.Column():
                with gr.Row():
                    gen_image = gr.Image(
                        label="Generated Image", interactive=False)

                    bg_removed_image = gr.Image(
                        label="bg removed image", interactive=False)

                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(
                        label="Output Model (OBJ Format)",
                        interactive=False,
                    )
                    gr.Markdown(
                        "Note: The model shown here is flipped. Download to get correct results.")
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(
                        label="Output Model (GLB Format)",
                        interactive=False,
                    )
                    gr.Markdown(
                        "Note: The model shown here has a darker appearance. Download to get correct results.")

        submit.click(fn=img_generator,
                     inputs=[input_image, text_prompt],
                     outputs=[gen_image],
                     ).success(
            fn=background_remover_func,
            inputs=[gen_image, do_remove_background, foreground_ratio],
            outputs=[bg_removed_image],
        ).success(
            fn=generate,
            inputs=[bg_removed_image, mc_resolution],
            outputs=[output_model_obj, output_model_glb],
        )

    with gr.Tab("VQI"):
        with gr.Column():
            with gr.Row():
                src = gr.Image(
                    label="src Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                bg_src_removed_image = gr.Image(
                    label="bg src removed image", interactive=False)

                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                    )
                    mc_resolution = gr.Slider(
                        label="Marching Cubes Resolution",
                        minimum=32,
                        maximum=320,
                        value=256,
                        step=32
                    )
                    convert_src = gr.Button("Convert source")
                with gr.Tab("OBJ"):
                    output_model_obj_src = gr.Model3D(
                        label="src Output Model (OBJ Format)",
                        interactive=False,
                    )
                    gr.Markdown(
                        "Note: The model shown here is flipped. Download to get correct results.")
                with gr.Tab("GLB"):
                    output_model_glb_src = gr.Model3D(
                        label="src Output Model (GLB Format)",
                        interactive=False,
                    )

        with gr.Column():
            with gr.Row():
                trg = gr.Image(
                    label="trg Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                bg_trg_removed_image = gr.Image(
                    label="bg target removed image", interactive=False)
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                    )
                    mc_resolution = gr.Slider(
                        label="Marching Cubes Resolution",
                        minimum=32,
                        maximum=320,
                        value=256,
                        step=32
                    )
                    convert_trg = gr.Button("Convert target")

                with gr.Tab("OBJ"):
                    output_model_obj_trg = gr.Model3D(
                        label="target Output Model (OBJ Format)",
                        interactive=False,
                    )
                    gr.Markdown(
                        "Note: The model shown here is flipped. Download to get correct results.")
                with gr.Tab("GLB"):
                    output_model_glb_trg = gr.Model3D(
                        label="target Output Model (GLB Format)",
                        interactive=False,
                    )

            b = gr.Button("Generate Analysis")

            with gr.Row():
                g = gr.Gallery(label="analysis", preview=True)
                issues = gr.Textbox(label="Issues")

            convert_src.click(
                fn=background_remover_func,
                inputs=[src],
                outputs=[bg_src_removed_image],
            ).success(
                fn=generate,
                inputs=[bg_src_removed_image],
                outputs=[output_model_obj_src, output_model_glb_src],
            )

            convert_trg.click(
                fn=background_remover_func,
                inputs=[trg],
                outputs=[bg_trg_removed_image],
            ).success(
                fn=generate,
                inputs=[bg_trg_removed_image],
                outputs=[output_model_obj_trg, output_model_glb_trg],
            )

            b.click(fake_gan, [src], [g])
            g.select(display, [g], issues)

if __name__ == '__main__':
    interface.launch(
        share=True,
        debug=True
    )
