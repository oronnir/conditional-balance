""" 
A script for generating style aligned artistic images by applying the Conditional-Balanced method
over the StyleAligned method using a content image with canny based control-net.
"""

import os
import argparse

import torch
from PIL import Image

from utils import generate_file_name, get_canny_ctrl, load_control_imgs, initialize_latents_controlnet, save_control_imgs, prepare_target_prompts
from models import create_infer_model, load_controlnet, load_ddpm


NUM_TIMESTEPS = 1000
NUM_SELF_ATT_LAYERS = 70
CONTROL_TYPE = 'canny'

EXTRA_STRONG_CANNY_PARAMS = (20, 40)
STRONG_CANNY_PARAMS = (50, 100)
MEDIUM_CANNY_PARAMS = (130, 200)
WEAK_CANNY_PARAMS = (300, 500)

def generate_canny_control_image(init_index, lambda_s, lambda_t, output_path, control_image_paths, style_prompt, reference_prompt, content_prompts, content_image_path=None, seed=42, canny_lower_t=.57, canny_upper_t=.8, no_control_output=None, num_images_per_prompt=1, num_inference_steps=50, initialize_latents=False):
    """
    Generate a canny control image from the given image path using the specified canny parameters.
    """
    device = 'cuda'
    num_style_layers = int(lambda_s)
    controlnet_removal_amount = int(lambda_t * NUM_TIMESTEPS)
    # Load DDPM and Handler
    controlnet = load_controlnet(CONTROL_TYPE, device=device)
    pipeline = load_ddpm(controlnet, device=device)
    infer_pipeline = create_infer_model(pipeline, num_style_layers=num_style_layers, controlnet_model=True, controlnet_removal_amount=controlnet_removal_amount)
    
    # Init output path
    output_path = os.path.join(output_path, CONTROL_TYPE, str(seed))
    os.makedirs(output_path, exist_ok=True)

    # Generate Control Image
    controlnet_conditioning_scale = 0.8
    if control_image_paths:
        control_img = load_control_imgs(control_image_paths)
    else:
        control_img = get_canny_ctrl(content_image_path, (canny_lower_t, canny_upper_t))
    num_controls = len(control_img) if type(control_img) is list else 1

    if not no_control_output:
        save_control_imgs(control_img, output_path)

    # Initialize DDPM parameters
    torch.manual_seed(seed)
    target_prompts = [f"{content_prompt}, {style_prompt}" for content_prompt in content_prompts]
    target_prompts = prepare_target_prompts(target_prompts, num_controls)
    reference_prompt = f"{reference_prompt}, {style_prompt}"
    prompts = [reference_prompt] + target_prompts

    print(f"Prompts: {prompts}")

    latents = None
    if initialize_latents:
        latents = initialize_latents_controlnet(num_images_per_prompt, num_controls, pipeline.unet.dtype)
    
    # Run Pipeline
    images, latents = infer_pipeline(prompts,
                            control_image=control_img,
                            num_inference_steps=num_inference_steps,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            num_images_per_prompt=num_images_per_prompt,
                            latents=latents)

    artist = style_prompt.split("in the style of")[-1].strip().replace(" ", "_")
    # Save Output
    f_name = generate_file_name(num_style_layers, controlnet_removal_amount=controlnet_removal_amount, content_image_path=None, seed=seed)
    images[0].save(os.path.join(output_path, f"ref_{artist}_{seed}.png"))
    # save latentsmass-assign
    torch.save(latents[0], os.path.join(output_path, f"ref_{artist}_{seed}.pt"))
    for i in range(1, len(images)):
        images[i].save(os.path.join(output_path, f"{init_index+i}_{artist}_" + f_name))
        # save latents
        torch.save(latents[i], os.path.join(output_path, f"{init_index+i}_{artist}_" + f_name + ".pt"))

def main(args):
    device = 'cuda'

    # Load DDPM and Handler
    controlnet = load_controlnet(CONTROL_TYPE, device=device)
    pipeline = load_ddpm(controlnet, device=device)
    infer_pipeline = create_infer_model(pipeline, num_style_layers=args.num_style_layers, controlnet_model=True, controlnet_removal_amount=args.controlnet_removal_amount)
    
    # Init output path
    args.output_path = os.path.join(args.output_path, CONTROL_TYPE, str(args.seed))
    os.makedirs(args.output_path, exist_ok=True)

    # Generate Control Image
    controlnet_conditioning_scale = 0.8
    if args.control_image_path:
        control_img = load_control_imgs(args.control_image_path)
    else:
        control_img = get_canny_ctrl(args.content_image_path, (args.canny_lower_t, args.canny_upper_t))
    num_controls = len(control_img) if type(control_img) is list else 1

    if not args.no_control_output:
        save_control_imgs(control_img, args.output_path)

    # Initialize DDPM parameters
    torch.manual_seed(args.seed)
    target_prompts = [f"{content_prompt}, {args.style_prompt}" for content_prompt in args.content_prompts]
    target_prompts = prepare_target_prompts(target_prompts, num_controls)
    reference_prompt = f"{args.reference_prompt}, {args.style_prompt}"
    prompts = [reference_prompt] + target_prompts

    print(f"Prompts: {prompts}")

    latents = None
    if args.initialize_latents:
        latents = initialize_latents_controlnet(args.num_images_per_prompt, num_controls, pipeline.unet.dtype)
    
    # Run Pipeline
    images, latents = infer_pipeline(prompts,
                            control_image=control_img,
                            num_inference_steps=args.num_inference_steps,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            num_images_per_prompt=args.num_images_per_prompt,
                            latents=latents)

    artist = args.style_prompt.split("in the style of")[-1].strip().replace(" ", "_")
    # Save Output
    f_name = generate_file_name(args.num_style_layers, controlnet_removal_amount=args.controlnet_removal_amount, content_image_path=None, seed=args.seed)
    images[0].save(os.path.join(args.output_path, f"ref_{artist}_{args.seed}.png"))
    # save latentsmass-assign
    torch.save(latents[0], os.path.join(args.output_path, f"ref_{artist}_{args.seed}.pt"))
    for i in range(1, len(images)):
        images[i].save(os.path.join(args.output_path, f"{i}_{artist}_" + f_name))
        # save latents
        torch.save(latents[i], os.path.join(args.output_path, f"{i}_{artist}_" + f_name + ".pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed")
    parser.add_argument('--content_prompts', nargs='+', type=str, help='List of prompts to generate')
    parser.add_argument("--style_prompt", type=str)
    parser.add_argument("--reference_prompt", type=str)
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--control_image_path", default=None)
    parser.add_argument("--canny_lower_t", type=int, default=MEDIUM_CANNY_PARAMS[0])
    parser.add_argument("--canny_upper_t", type=int, default=MEDIUM_CANNY_PARAMS[1])
    parser.add_argument("--no_control_output", action="store_true")
    parser.add_argument("--lambda_s", default=0.43, type=float)
    parser.add_argument("--lambda_t", default=0.85, type=float)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--initialize_latents", action='store_true')
    parser.add_argument("--output_path")
    args = parser.parse_args()
    
    assert 0 <= args.lambda_s <= NUM_SELF_ATT_LAYERS, "Plase enter a float value in range [0, 1] (ratio) or an int value in range [0, 70] (number of layers for stylization)"
    if 0 < args.lambda_s <= 1:
        args.lambda_s = args.lambda_s * NUM_SELF_ATT_LAYERS
    args.num_style_layers = int(args.lambda_s)

    assert 0 <= args.lambda_t <= NUM_TIMESTEPS, "Plase enter a float value in range [0, 1] for setting the geometric style ration (0-no geometric style, 1-full geometric style)"
    args.controlnet_removal_amount = int(args.lambda_t * NUM_TIMESTEPS)

    assert (args.content_image_path is None) ^ (args.control_image_path is None), "Please provide exactly one of the following params: content_image_path, control_image_path"
    main(args)
