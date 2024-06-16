import sys
sys.path.insert(0, "./diffusers/src")

import torch
from diffusers import StableDiffusion3Pipeline
import time

print("begin")

'''
pipe = StableDiffusion3Pipeline.from_pretrained(
    "./stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    text_encoder_3=None)
'''

pipe = StableDiffusion3Pipeline.from_single_file(
    #"./stable-diffusion-3-medium/sd3_medium_incl_clips.safetensors",
    "./stable-diffusion-3-medium/sd3_medium_incl_clips_t5xxlfp8.safetensors",
    config="./stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    text_encoder_3=None
)


if (sys.argv[1] == "cuda"):
    pipe = pipe.to("cuda")
else:
    pipe.enable_model_cpu_offload()


while (1):
    prompt = input("> prompt('bye' to exit): ")
    if (prompt == "bye"):
        break
    
    steps = input("> steps(default 20): ")
    if (steps.isdigit()):
        steps = int(steps)
    else:
        steps = 20
    print("steps = {}".format(steps))

    pic_num = input("> pic_num(default 1): ")
    if (pic_num.isdigit()):
        pic_num = int(pic_num)
    else:
        pic_num = 1
    print("pic_num = {}".format(pic_num))

    prompt_list = [prompt for _ in range(pic_num)]

    print("processing...")

    start_tv = time.time()
    images = pipe(
        prompt_list,
        negative_prompt="",
        num_inference_steps=steps,
        guidance_scale=7.0,
    ).images
    end_tv = time.time()
    print("done, time cost={}s".format(end_tv - start_tv))

    for i, image in enumerate(images):
        pic_name = "sd3_out{}.png".format(i)
        print("{} saved!".format(pic_name))
        image.save(pic_name)

print("finished")