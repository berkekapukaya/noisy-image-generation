from diffusers import AutoPipelineForText2Image
import torch
import random

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    variant="fp16")

pipeline = pipeline.to("mps")

pipeline.enable_attention_slicing()

i = 0

while i < 1:

    randSeed = random.randint(1, 1000)

    generator = torch.Generator(device="mps").manual_seed(randSeed)

   ##30 inference steps blurry

    # image = pipeline(
    #     prompt="Astronaut in a jungle, cold color palette, muted colors, noisy, blurry, loss of sharpness",
    #     negative_prompt="ugly, deformed, disfigured, bad anatomy, sharp, high quality, high detail",
    #     generator=generator, num_inference_steps=30,
    # ).images[0]
    #
    # image.save("text2img/blurry/30-steps/stained_glass%s.png" % randSeed)


    ##50 inference blurry uncomment the preferred one
    """
    image = pipeline(
        prompt="Astronaut in a jungle, cold color palette, muted colors, noisy, blurry, loss of sharpness",
        negative_prompt="ugly, deformed, disfigured, bad anatomy, sharp, high quality, high detail",
        generator=generator,
    ).images[0]

    image.save("text2img/blurry/50-steps/stained_glass%s.png" % randSeed)
    """

 ##30 inference steps foggy
    """
    image = pipeline(
        prompt="a person on the street in fog, cold color palette, foggy, fog on the foreground, muted colors, blurry, loss of sharpness",
        negative_prompt="ugly, deformed, disfigured, bad anatomy, clear, sharp, high quality, high detail",
        generator=generator, num_inference_steps=30,
    ).images[0]

    image.save("text2img/foggy/30-steps/stained_glass%s.png" % randSeed)"""

    """
    ##50 inference foggy uncomment the preferred one
    image = pipeline(
        prompt="a person on the street in fog, cold color palette, foggy, fog on the foreground, muted colors, blurry, loss of sharpness",
        negative_prompt="ugly, deformed, disfigured, bad anatomy, sharp, high quality, high detail",
        generator=generator,
    ).images[0]

    image.save("text2img/foggy/50-steps/stained_glass%s.png" % randSeed)
    """

    ## 30 inference steps darkened

    # image = pipeline(
    #     prompt="image of an alley shot during the night, loss of light, low brightness, low contrast",
    #     negative_prompt="ugly, deformed, disfigured, bad anatomy, sharp, high quality, high detail",
    #     generator=generator, num_inference_steps=30,
    # ).images[0]
    #
    # image.save("text2img/darkened/30-steps/darkened_alley%s.png" % randSeed)

    ## 50 inference steps darkened

    image = pipeline(
        prompt="image of an alley during the night, no lighting, loss of light, low brightness, loss in detail",
        negative_prompt="ugly, deformed, disfigured, bad anatomy, sharp, high quality, high detail",
        generator=generator, num_inference_steps=50,
    ).images[0]

    image.save("text2img/darkened/50-steps/darkened_alley%s.png" % randSeed)

    i += 1



