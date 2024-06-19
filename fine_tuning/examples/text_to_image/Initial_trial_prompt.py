from diffusers import StableDiffusionPipeline
import torch
import random

model_path = "/home/berke/fine_tuned_diffusing/diffusers/examples/text_to_image/sd-naruto80000-model-lora" # Your model pathq
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")


prompt = "naruto, naruto style, anime like drawings, japanese art"
negative_prompt = "bad anatomy, disfigured, ugly, deformed, poor details, disfigured face"

i = 0

while i < 15:

    randSeed = random.randint(1, 1000)

    generator = torch.Generator(device="cuda").manual_seed(randSeed)

    # pass prompt and image to pipeline
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator,
                 num_inference_steps=75, guidance_scale=7.5).images[0]
    image.save("/home/berke/fine_tuned_diffusing/diffusers/examples/text_to_image/lora_images/yoda_80000/yoda%s.png" % randSeed)

    i += 1
