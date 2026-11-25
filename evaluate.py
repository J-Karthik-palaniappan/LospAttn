#====================================
# PATHS
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

DIT256_MODEL = "facebook/DiT-XL-2-256"
DIT512_MODEL = "facebook/DiT-XL-2-512"
PIX1K_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
PIX2K_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS"

PIXART_BASE = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"

#====================================
# LOAD MODELS
import torch
from huggingface_hub import snapshot_download
from diffusers import DiTPipeline, Transformer2DModel, PixArtSigmaPipeline

def load_dit(model, dtype=torch.float16, device="cuda", cache_dir=None):
    model_paths = {"dit256":DIT256_MODEL, "dit512":DIT512_MODEL}
    model_repo = model_paths[model]
    # Download models if not present
    model_path = snapshot_download(repo_id=model_repo, cache_dir=cache_dir,)
    pipe = DiTPipeline.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True).to(device)
    print("Scheduler:", type(pipe.scheduler).__name__)
    return pipe

def load_pixart(model, dtype=torch.float16, device="cuda", cache_dir=None):
    model_paths = {"pixart1k": PIX1K_MODEL, "pixart2k": PIX2K_MODEL,}
    model_repo = model_paths[model]
    # Download models if not present
    model_path = snapshot_download(repo_id=model_repo, cache_dir=cache_dir,)
    base_path = snapshot_download(repo_id=PIXART_BASE, cache_dir=cache_dir,)

    # Load common transformer architecture with model-specific weights
    transformer = Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype,
        use_safetensors=True, local_files_only=True,
    )
    # Load common pipeline components
    pipe = PixArtSigmaPipeline.from_pretrained(
        base_path, transformer=transformer, torch_dtype=dtype, 
        use_safetensors=True, local_files_only=True,
    )
    pipe.config._name_or_path = model_repo
    pipe.to(device)
    print("Scheduler:", type(pipe.scheduler).__name__)
    return pipe

#====================================
# TEST SCRIPTS
import os

def test_dit(pipe, class_labels, save_path=None, n_steps=50, 
            device="cuda", seed=0, cfg_scale=4.0, batch=8):
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(seed)  
    l = len(class_labels)
    for i in range(0, l, batch):
        print(f"batch:{i}")
        with torch.no_grad():
            images = pipe(
                class_labels=class_labels[i: min(i+batch, l)],
                num_inference_steps=n_steps,
                guidance_scale=cfg_scale,
                generator=generator,
            ).images
        
        if save_path is not None:
            for j, image in enumerate(images):
                image.save(f"{save_path}/{i+j}.jpg")
                

def test_pixart(pipe, prompts, save_path=None, n_steps=50, 
        device="cuda", seed=0, cfg_scale=4.5, batch=4):
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(seed)  
    l = len(prompts)
    for i in range(0, l, batch):
        print(f"batch:{i}")
        with torch.no_grad():
            images = pipe(
                prompts[i: min(i+batch, l)],
                num_inference_steps=n_steps,
                guidance_scale=cfg_scale,
                generator=generator,
            ).images
        
        if save_path is not None:
            for j, image in enumerate(images):
                image.save(f"{save_path}/{i+j}.jpg")

#====================================
# EXTRAS
CALIB_LABELS = [
        681, 106, 172, 251, 64, 700, 747, 847, 118, 965,
        645, 574, 458, 6, 949, 542, 238, 504, 500, 66,
        510, 815, 4, 852, 798, 282, 314, 171, 278, 297,
        841, 319, 367, 443, 89, 141, 875, 695, 234, 393,
        193, 122, 438, 910, 125, 332, 273, 669, 729, 9
    ]
TEST_LABELS = [420, 780, 356, 456, 726, 290, 424, 0]

CALIB_PROMPTS = [
    "A giraffe looking and  walking around a field.",
    "White toilet in a small tiled room rainbows on toilet",
    "A black bear walks down a hill away from a tree.",
    "sheep running through town all together in a crowd",
    "older man tries to block younger man's frisbee throw",
    "A city bus that is traveling down a wet country road.",
    "Parking meter street side looking into a fenced parking lot.",
    "Several people holding umbrella's riding in a boat.",
    "A guy tossing a tennis ball up and preparing to serve.",
    "A blond girl holding a baseball bat and smiling with her hand over her mouth.",
    "A baby is looking at a book while sucking a pacifier.",
    "Adult sitting on wooden park bench in large open space.",
    "A black bag attached to the long handle of a piece of luggage.",
    "The woman in the diner and the man looking into the window are making eye contact.",
    "A person with a tie and a suit.",
    "Three bowls filled with vegetables on top of a table."
]
        
TEST_PROMPTS = [
    "A man riding a bicycle down a city street.",
    "A woman walking a dog on a sidewalk.",
    "A group of people sitting at a dining table.",
    "A brown dog running through a grassy field.",
    "A cat sitting on a windowsill.",
    "A red bus driving along a road.",
    "A young child playing with a ball in a park.",
    "Two people riding horses across a field.",
    "A person surfing on a wave.",
    "A couple of people skiing down a snowy slope.",
    "A large airplane flying through a cloudy sky.",
    "A boat floating on a calm lake.",
    "A train traveling along railway tracks.",
    "A motorcycle parked beside a car.",
    "A man holding an umbrella on a rainy street.",
    "A woman carrying a handbag while crossing the street.",
    "Several people waiting at a bus stop.",
    "A chef preparing food in a kitchen.",
    "A plate of food on a wooden table.",
    "A person eating pizza at a restaurant.",
    "A bowl of fruit sitting on a kitchen counter.",
    "A dog catching a frisbee in a park.",
    "A flock of birds flying over the water.",
    "An elephant walking across a grassy savanna.",
    "A giraffe standing beside a tree.",
    "A zebra grazing in a field.",
    "A bear standing near a river.",
    "A group of sheep grazing on a hillside.",
    "A cow standing in a green pasture.",
    "A horse eating grass near a fence.",
    "A person playing tennis on an outdoor court.",
    "A basketball player shooting toward the hoop.",
    "A soccer player kicking a ball on the field.",
    "A baseball player holding a bat at home plate.",
    "A person riding a skateboard on a ramp.",
    "A man rowing a boat on a river.",
    "A woman taking a photograph with a camera.",
    "A person reading a book on a couch.",
    "A laptop computer sitting on a desk.",
    "A television mounted on a living room wall.",
    "A clock hanging above a doorway.",
    "A vase of flowers sitting on a table.",
    "A street sign standing beside a road.",
    "A fire hydrant on a city sidewalk.",
    "A stop sign at a quiet intersection.",
    "A parked bicycle leaning against a wall.",
    "A refrigerator standing in a modern kitchen.",
    "A bed with pillows in a bright bedroom.",
    "A suitcase resting on an airport floor.",
    "A person walking through a crowded market.",
]