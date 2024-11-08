<<<<<<< HEAD
#coorect code
import os
import torch
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from PIL import Image
import ngrok
import uvicorn
import nest_asyncio

nest_asyncio.apply()


class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16, variant='fp16'
).to(CFG.device)

description_model = pipeline("text-generation", model=CFG.prompt_gen_model_id)

ngrok.set_auth_token("2kCSn0uFjMG24Kz87dUKXskSgBz_7JHf7aJv5wCVDBkPUAh6Q")


listener = ngrok.forward("127.0.0.1:5000", authtoken_from_env=True, domain="active-mallard-uniquely.ngrok-free.app")



class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate_image_and_description")
async def generate_image_and_description(request: ImageRequest):
    prompt = request.prompt

    image = image_gen_model(prompt, num_inference_steps=CFG.image_gen_steps, guidance_scale=CFG.image_gen_guidance_scale).images[0]

    image = image.resize(CFG.image_gen_size, Image.LANCZOS)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    description_prompt = f"Describe the following image: {prompt}"
    description = description_model(description_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']

    return {"image": img_str, "description": description}

if __name__ == "__main__":
    public_url = ngrok.connect(5000)
    print(f"Public URL: {public_url}")

=======
#coorect code
import os
import torch
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from PIL import Image
import ngrok
import uvicorn
import nest_asyncio

nest_asyncio.apply()


class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16, variant='fp16'
).to(CFG.device)

description_model = pipeline("text-generation", model=CFG.prompt_gen_model_id)

ngrok.set_auth_token("2kCSn0uFjMG24Kz87dUKXskSgBz_7JHf7aJv5wCVDBkPUAh6Q")


listener = ngrok.forward("127.0.0.1:5000", authtoken_from_env=True, domain="active-mallard-uniquely.ngrok-free.app")



class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate_image_and_description")
async def generate_image_and_description(request: ImageRequest):
    prompt = request.prompt

    image = image_gen_model(prompt, num_inference_steps=CFG.image_gen_steps, guidance_scale=CFG.image_gen_guidance_scale).images[0]

    image = image.resize(CFG.image_gen_size, Image.LANCZOS)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    description_prompt = f"Describe the following image: {prompt}"
    description = description_model(description_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']

    return {"image": img_str, "description": description}

if __name__ == "__main__":
    public_url = ngrok.connect(5000)
    print(f"Public URL: {public_url}")

>>>>>>> 5403f3ca35d074857667065c608d6859bbc36283
    uvicorn.run(app, host="0.0.0.0", port=5000)