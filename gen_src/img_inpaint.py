import base64
import io

import torch
from diffusers import AutoPipelineForInpainting, LCMScheduler
from diffusers.utils import load_image, make_image_grid
from fastapi import FastAPI, Request
from fastapi.responses import Response

pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")


# set scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# load LCM-LoRA
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")


def inpaint_img(init_image, mask_image, prompt="A radio transistor"):

    generator = torch.manual_seed(99999)
    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        generator=generator,
        num_inference_steps=8,
        guidance_scale=1,
    ).images[0]
    return image


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/image_inpaint")
async def inpaint_img_func(request: Request):
    json_obj = await request.json()
    init_image = json_obj["init_image"]
    mask_image = json_obj["mask"]
    prompt = json_obj["prompt"]

    image = inpaint_img(init_image=init_image,
                        mask_image=mask_image,
                        prompt=prompt)
    byte_space = io.BytesIO()
    image.save(byte_space, "png")
    return Response(content=byte_space.getvalue(),
                    media_type="image/png")

# # uncomment this in colab
# import nest_asyncio
# from pyngrok import ngrok
# import uvicorn
# ngrok.set_auth_token("2iNsGhWfDRuFD2HQtN5N0MnE3HP_2vYXmk3mZXk1AogR7KJrK")
# ngrok_tunnel = ngrok.connect(8000)
# print('Public URL:', ngrok_tunnel.public_url)
# nest_asyncio.apply()
# uvicorn.run(app, port=8000)
