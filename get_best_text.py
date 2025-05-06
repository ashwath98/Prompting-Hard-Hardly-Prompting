import open_clip
import torch
from PIL import Image
import argparse
import os

# Add command-line argument parsing
parser = argparse.ArgumentParser(description='Find the best text prompt matching an image using CLIP similarity')
parser.add_argument('--prompt_file', type=str, required=True, help='Path to file containing candidate prompts')
parser.add_argument('--image_path', type=str, required=True, help='Path to the reference image')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output image and results')
parser.add_argument('--output_image_id', type=str, required=True, help='Name of the output image')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)

from diffusers import StableDiffusionPipeline
from diffusers import PNDMScheduler

def measure_clip_similarity(orig_images, pred_images, clip_model, device):
    with torch.no_grad():
        orig_feat = clip_model.encode_image(orig_images)
        orig_feat = orig_feat / orig_feat.norm(dim=1, keepdim=True)

        pred_feat = clip_model.encode_image(pred_images)
        pred_feat = pred_feat / pred_feat.norm(dim=1, keepdim=True)
        return (orig_feat @ pred_feat.t()).mean().item()

model_id = "runwayml/stable-diffusion-v1-5"
scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir='.')

weight_dtype = torch.float16

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=weight_dtype, cache_dir='.')
pipe = pipe.to(device)
image_length = 512

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

best_loss=0.
step =0
orig_image = Image.open(args.image_path).convert('RGB')
with open(args.prompt_file, 'r') as textfile:
    prompt = textfile.readlines()
    for prompt_l in prompt:
        step=step+1
        if step % 1 == 0:
            with torch.no_grad():
                pred_imgs = pipe(
                    prompt_l,
                    num_images_per_prompt=1,
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    height=image_length,
                    width=image_length,
                    ).images
                orig_images_temp = [clip_preprocess(orig_image).unsqueeze(0)]
                orig_images_t = torch.cat(orig_images_temp).to(device)
                pred_imgs_temp = [clip_preprocess(i).unsqueeze(0) for i in pred_imgs]
                pred_imgs_t = torch.cat(pred_imgs_temp).to(device)
                eval_loss = measure_clip_similarity(orig_images_t, pred_imgs_t, clip_model, device)
                print(step)

                if best_loss < eval_loss:
                    best_loss = eval_loss
                    best_text = prompt_l
                    best_pred = pred_imgs[0]

output_image_path = os.path.join(args.output_dir, args.output_image_id+'.png')
best_pred.save(output_image_path)
print()
print(f"Best shot: cosine similarity: {best_loss:.3f}")
print(f"text: {best_text}")
best_text_file = os.path.join(args.output_dir, args.output_image_id+'.txt')
with open(best_text_file, 'w') as f:
    f.write(best_text)
print(f"Saved best image to: {output_image_path}")
