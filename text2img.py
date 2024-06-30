import base64
import io
import torch
from diffusers import DiffusionPipeline, LCMScheduler
from fastapi import FastAPI, Request
from fastapi.responses import Response
# from PIL import Image

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16
).to("cuda")

# set scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# load LCM-LoRA
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")


def generate_text2img(prompt="A radio transistor"):
    # image = Image.open("self-image.jpeg")
    # return image
    generator = torch.manual_seed(42)
    image = pipe(
        prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=1.0
    ).images[0]
    return image


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/gentext2img")
async def text2img_func(request: Request):
    json_obj = await request.json()
    prompt = json_obj["prompt"]
    image = generate_text2img(prompt)
    byte_space = io.BytesIO()
    image.save(byte_space, "png")
    return Response(content=byte_space.getvalue(),
                    media_type="image/png")
