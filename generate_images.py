import argparse
import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, PNDMScheduler

def generate_images_from_prompts(prompts_file, output_dir, model_id="runwayml/stable-diffusion-v1-5", 
                                cache_dir='.', device=None, image_length=512, 
                                guidance_scale=7.5, num_inference_steps=50):
    """Generate images from prompts in a file and save them to output_dir"""
    
    # Set up device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up model
    print(f"Loading Stable Diffusion model: {model_id}")
    scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=cache_dir)
    weight_dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
        cache_dir=cache_dir
    )
    pipe = pipe.to(device)
    print("Model loaded successfully")
    
    # Read prompts from file
    print(f"Reading prompts from {prompts_file}")
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(prompts)} prompts")
    
    # Generate and save images
    image_paths = []
    with open(os.path.join(output_dir, "prompts_used.txt"), 'w') as log_file:
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}: '{prompt[:50]}...'")
            log_file.write(f"Image {i:04d}: {prompt}\n")
            
            # Generate image
            with torch.no_grad():
                image = pipe(
                    prompt,
                    num_images_per_prompt=1,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=image_length,
                    width=image_length,
                ).images[0]
            
            # Save image
            image_path = os.path.join(output_dir, f"image_{i:04d}.png")
            image.save(image_path)
            image_paths.append(image_path)
            print(f"  Saved to {image_path}")
    
    return image_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from text prompts using Stable Diffusion")
    parser.add_argument("--prompts_file", required=True, help="Path to file containing prompts (one per line)")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated images")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model ID")
    parser.add_argument("--cache_dir", default=".", help="Model cache directory")
    parser.add_argument("--device", help="Device to use (e.g., 'cuda', 'cpu'). Auto-detects if not specified")
    parser.add_argument("--image_length", type=int, default=512, help="Size of generated images")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for diffusion")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    
    args = parser.parse_args()
    
    generate_images_from_prompts(
        args.prompts_file,
        args.output_dir,
        args.model_id,
        args.cache_dir,
        args.device,
        args.image_length,
        args.guidance_scale,
        args.num_inference_steps
    ) 