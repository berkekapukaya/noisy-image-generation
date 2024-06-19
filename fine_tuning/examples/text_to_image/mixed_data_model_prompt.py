from diffusers import StableDiffusionPipeline
import torch
import random

model_path = "/home/berke/fine_tuned_diffusing/diffusers/examples/text_to_image/sd-low-light-model-lora" # Your model path
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")


prompt = ("a forest at midnight, "
           "darkness, pitch black,"
           "shot with dslr, low exposure, low-light,"
           "night time, low brightness,"
           "loss of detail, 4k")
negative_prompt = "bad anatomy, disfigured, ugly, deformed, poor details, disfigured face"

i = 0

while i < 15:

    randSeed = random.randint(1, 10000)

    generator = torch.Generator(device="cuda").manual_seed(randSeed)

    # pass prompt and image to pipeline
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator,
                 num_inference_steps=75, guidance_scale=7.5).images[0]
    image.save("/home/berke/fine_tuned_diffusing/diffusers/examples"
               "/text_to_image/lora_images/mixed_Dataset_80000/a_room/yoda%s.png" % randSeed)

    i += 1
