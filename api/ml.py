import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image
from pydantic import BaseSettings


class Settings(BaseSettings):
    HUGGING_FACE_TOKEN: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


env = Settings()

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=env.HUGGING_FACE_TOKEN,
)

pipe.to("cuda")


def obtain_image(
    prompt: str,
    *,
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> Image:
    generator = None if seed is None else torch.Generator("cuda").manual_seed(seed)
    print(f"Using device: {pipe.device}")
    image: Image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    return image


# image = obtain_image(
#     prompt="a photograph of an astronaut riding a horse",
#     num_inference_steps=5,
#     seed=1024,
# )
