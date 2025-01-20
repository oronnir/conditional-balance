""" 
A script for generating style aligned artistic images by applying the Conditional-Balanced method
over the StyleAligned method.
"""

import os
import argparse
import torch

from utils import generate_file_name
from models import create_infer_model, load_ddpm

NUM_SELF_ATT_LAYERS = 70

def main(args):
    device = 'cuda'

    # Load DDPM and Handler
    pipeline = load_ddpm(device=device)
    infer_pipeline = create_infer_model(pipeline, args.num_style_layers, controlnet_model=False)
    
    # Init output path
    args.output_path = os.path.join(args.output_path, 'text', str(args.seed))
    os.makedirs(args.output_path, exist_ok=True)

    # Initialize DDPM parameters
    torch.manual_seed(args.seed)
    target_prompt = f"{args.content_prompt}, {args.style_prompt}"
    reference_prompt = f"{args.reference_prompt}, {args.style_prompt}"

    latents = None
    if args.initialize_latents:
        latents = torch.randn(2, 4, 128, 128).to(pipeline.unet.dtype)
        latents[1:] = torch.randn(1, 4, 128, 128).to(pipeline.unet.dtype)

        additional_latents = []
        if args.num_output_imgs > 1:
            for _ in range(args.num_output_imgs - 1):
                additional_latents.append(torch.randn(1, 4, 128, 128).to(pipeline.unet.dtype))
            latents = torch.cat([latents, torch.cat(additional_latents, dim=0)], dim=0)
    
    # Run Pipeline
    images = infer_pipeline([reference_prompt, target_prompt],
                            num_inference_steps=args.num_inference_steps,
                            num_images_per_prompt=args.num_output_imgs,
                            latents=latents)
    
    # Save Output
    f_name = generate_file_name(args.num_style_layers, controlnet_removal_amount=0, content_image_path=None, seed=args.seed)
    images[0].save(os.path.join(args.output_path, f"ref_{args.seed}.png"))
    for i in range(args.num_output_imgs):
        images[i+1].save(os.path.join(args.output_path, f"{i}_" + f_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed")
    parser.add_argument("--content_prompt", type=str)
    parser.add_argument("--style_prompt", type=str)
    parser.add_argument("--reference_prompt", type=str)
    parser.add_argument('--lambda_s', default=0.43, type=float)
    parser.add_argument("--num_output_imgs", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--initialize_latents", action='store_true')
    parser.add_argument('--output_path', default='./outputs')
    args = parser.parse_args()

    
    assert 0 <= args.lambda_s <= NUM_SELF_ATT_LAYERS, "Plase enter a float value in range [0, 1] (ratio) or an int value in range [0, 70] (number of layers for stylization)"
    if 0 < args.lambda_s <= 1:
        args.num_style_layers = args.lambda_s * NUM_SELF_ATT_LAYERS
    args.num_style_layers = int(args.lambda_s) 

    assert args.num_output_imgs > 0, f"'num_output_imgs' should be at least 1. Current value: {args.num_output_images}"

    main(args)

