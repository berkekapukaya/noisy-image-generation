import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import random

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, variant="fp16", use_safetensors=True
)

pipeline = pipeline.to("mps")

pipeline.enable_attention_slicing()

# prepare image noisy

url ="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
init_image = load_image('../img2img_samples/balloons_noisy.png')

prompt = "an image of a street in the city, noisy, colored, photorealistic, noisy image"
negative_prompt = "bad anatomy, disfigured, ugly, deformed, poor details, disfigured face"


# image foggy
# url = "https://www.ignatianspirituality.com/wp-content/uploads/2015/06/man-in-fog2.jpg"
# init_image = load_image(url)
#
#
# prompt = "a person in fog, fog on the foreground, photorealistic, 8k"
# prompt_alt = "an image of a foggy forest, fog on the foreground, photorealistic, 8k"
# prompt_alt2 = "an image of a foggy alley in the city, fog on the foreground, photorealistic, 8k"
# negative_prompt = "bad anatomy, disfigured, ugly, deformed, poor details, disfigured face"

# image darkened

# url = "../img2img_samples/dark_forest.jpg"
# init_image = load_image(url)
#
#
# prompt = "an image of a forest shot in the dark, night time, low light, loss of detail, photorealistic, 8k"
# prompt_alt = "an image of a foggy forest, fog on the foreground, photorealistic, 8k"
# negative_prompt = "bad anatomy, disfigured, ugly, deformed, poor details, disfigured face"

i = 0

while i < 15:

    randSeed = random.randint(1, 1000)

    generator = torch.Generator(device="mps").manual_seed(randSeed)

    # pass prompt and image to pipeline
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, generator=generator, image=init_image).images[0]
    image.save("../img2img/noisy_alley/nalley%s.png" % randSeed)

    i += 1