from pydantic import BaseModel, Field
from fal.toolkit import Image


class Text2ImgInput(BaseModel):
    prompt: str = Field(
        description="The prompt to generate an image from",
        examples=["A bunny playing in the grass"],
        min_length=1,
        max_length=1000,
    )
    negative_prompt: str = Field(
        description="The prompt to not generate",
        examples=["ugly, boring"],
        default="ugly, boring",
        max_length=1000,
    )
    height: int = Field(
        description="The height of the image you wish to generate, in pixels",
        default=1024,
        ge=256,
        le=2048,
    )
    width: int = Field(
        description="The width of the image you wish to generate, in pixels",
        default=1024,
        ge=256,
        le=2048,
    )
    guidance: float = Field(
        description="How closely the image generation should adhere to the prompt",
        examples=[7.5],
        default=7.5,
        ge=0.0,
    )
    steps: int = Field(
        description="How many diffusion steps to take",
        examples=[20],
        default=20,
        ge=1,
    )
    seed: int = Field(
        description="Random seed for image generation",
        examples=[42],
        default=42,
        ge=0,
        le=2147483647,
    )
    nag_scale: float = Field(
        description="NAG-PAG negative guidance scale. Higher values increase negative prompt influence.",
        examples=[0.1, 0.3, 0.5],
        default=0.3,
        ge=0.0,
    )

    # im unsure if i wanna let this be configurable, ill leave it as 0.5 tbh
    # alpha: float = Field(
    #     description="NAG-PAG blending factor (0.0-1.0). Controls mixing of positive and negative guidance.",
    #     examples=[0.7, 0.8, 0.9],
    #     default=0.8,
    #     ge=0.0,
    #     le=1.0,
    # )


class Img2ImgInput(BaseModel):
    image: Image = Field(
        description="The image to generate from",
    )
    prompt: str = Field(
        description="The prompt to generate an image from",
        examples=["A bunny playing in the grass"],
        min_length=1,
        max_length=1000,
    )
    height: int = Field(
        description="The height of the image you wish to generate, in pixels",
        default=1024,
        ge=256,
        le=2048,
    )
    width: int = Field(
        description="The width of the image you wish to generate, in pixels",
        default=1024,
        ge=256,
        le=2048,
    )
    guidance: float = Field(
        description="How closely the image generation should adhere to the prompt",
        examples=[3.5],
        default=3.5,
        ge=1.0,
        le=20.0,
    )
    strength: float = Field(
        description="The strength of the image to generate from, between 0 and 1",
        examples=[0.5],
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    steps: int = Field(
        description="How many diffusion steps to take",
        examples=[20],
        default=20,
        ge=1,
    )
    seed: int = Field(
        description="Random seed for image generation", examples=[42], default=42
    )


class ImgOutput(BaseModel):
    image: Image
