from diffusers import StableDiffusionPipeline
import torch
import random

model_path = "/home/berke/fine_tuned_diffusing/diffusers/examples/text_to_image/low_light_only_low_model_lora" # Your model path
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")


prompt0 = "a library with narrow corridors, low-light, darkness, pitch black, night time, low brightness, 4k"
prompt1 = ("a forest at midnight, "
           "darkness, pitch black,"
           "shot with dslr, low exposure, low-light,"
           "night time, low brightness,"
           "loss of detail, 4k")
negative_prompt = "bad anatomy, disfigured, ugly, deformed, high detail, high brightness disfigured face"

i = 0

while i < 15:

    randSeed = random.randint(1, 10000)

    generator = torch.Generator(device="cuda").manual_seed(randSeed)

    # pass prompt and image to pipeline
    image = pipe(prompt=prompt1, negative_prompt=negative_prompt, generator=generator,
                 num_inference_steps=50, guidance_scale=10.5).images[0]
    image.save("/home/berke/fine_tuned_diffusing/diffusers/examples"
               "/text_to_image/lora_images/mixed_Dataset_15000/forest_low_exposure%s.png" % randSeed)

    i += 1
