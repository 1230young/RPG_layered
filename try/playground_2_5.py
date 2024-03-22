from diffusers import DiffusionPipeline
import torch
from PIL import Image
from diffusers import EDMDPMSolverMultistepScheduler

pipe = DiffusionPipeline.from_pretrained(
    "/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/models/playground-2.5",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# # Optional: Use DPM++ 2M Karras scheduler for crisper fine details
# from diffusers import EDMDPMSolverMultistepScheduler
# pipe.scheduler = EDMDPMSolverMultistepScheduler()

prompts = ["Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
           "blurred landscape, close-up photo of man, 1800s, dressed in t-shirt",
           "human with pig head wearing streetwear fashion clothes, skateboarding on a skatepark",
           "Seasoned fisherman portrait, weathered skin etched with deep wrinkles, white beard, piercing gaze beneath a fisherman’s hat, softly blurred dock background accentuating rugged features, captured under natural light, ultra-realistic, high dynamic range phot",
           "tattoo sketch chicano style, beauty and pit bull, blood, dynamics",
           "Bilibin ink detailed masterpiece, National Geographic gracious, filigree acrylic, Slavic folklore, tender face, storybook illustration, art on a cracked wood, young beautiful ed Riding Hood girl, book illustration style, forest, mushrooms, hyperrealism, digital art, cinematic, close portrait, highly detailed expressive glowing eyes, airy, detailed face, shadow play, realistic textures, dynamic pose, unusual, modern. heartwarming, cozy, fairytale, fantasy, detailed textures, artistic dynamic pose, tender, atmospheric, sharp focus, centered composition, complex background, soft haze, masterpiece. animalistic, beautiful, tiny detailed",
           "image of a jade green and gold coloured Fabergé egg, 16k resolution, highly detailed, product photography, trending on artstation, sharp focus, studio photo, intricate details, fairly dark background, perfect lighting, perfect composition, sharp features, Miki Asai Macro photography, close-up, hyper detailed, trending on artstation, sharp focus, studio photo, intricate details, highly detailed, by greg rutkowski"
           ]
for n, prompt in enumerate(prompts):
    image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]
    image.save(f"try/{n}.png")



