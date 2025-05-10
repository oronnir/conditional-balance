import os
import numpy as np
import torch
import cv2
from PIL import Image

from transformers import DPTImageProcessor, DPTForDepthEstimation


def get_depth_ctrl(content_image_path, device):
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
    feature_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

    if os.path.isdir(content_image_path):
        input_imgs = [Image.open(os.path.join(content_image_path, file)).resize((1024, 1024)).convert("RGB") for file in os.listdir(content_image_path)]
        control_img = [calc_depth_map(img, feature_processor, depth_estimator) for img in input_imgs]
    else:
        input_img = Image.open(content_image_path).resize((1024, 1024)).convert("RGB")
        control_img = calc_depth_map(input_img, feature_processor, depth_estimator)
    return control_img

def get_canny_ctrl(content_image_path, canny_params):
    if os.path.isdir(content_image_path):
        input_imgs = [Image.open(os.path.join(content_image_path, file)).resize((1024, 1024)) for file in sorted(os.listdir(content_image_path))]
        control_img = [prepare_canny_image(img, canny_params=canny_params) for img in input_imgs]
    else:
        input_img = Image.open(content_image_path).resize((1024, 1024))
        control_img = prepare_canny_image(input_img, canny_params=canny_params)
    return control_img

def get_dummy_ctrl():
    return Image.fromarray(np.zeros((1024, 1024, 3)).astype(np.uint8))


def calc_depth_map(image: Image, feature_processor: DPTImageProcessor, depth_estimator: DPTForDepthEstimation) -> Image:
    image = feature_processor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def prepare_canny_image(image_pil, canny_params):
    image_pil = image_pil.convert("RGB")
    image = np.array(image_pil)

    image = cv2.Canny(image, *canny_params)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image

def generate_file_name(num_style_layers, controlnet_removal_amount, content_image_path, seed):
    starter = content_image_path.split('/')[-1].split('.')[0] if content_image_path else 'gen'
    addition = f"s{num_style_layers}"

    if controlnet_removal_amount:
        addition = addition + f"_t{controlnet_removal_amount}"

    f_name = f"{starter}_{addition}_{seed}.png"
    return f_name

def load_control_imgs(path):
    if type(path) is str and os.path.isdir(path):
        ctrl_img = [Image.open(os.path.join(path, file)) for file in os.listdir(path)]
    elif type(path) is list:
        print(f'\n\nFound {len(path)} control images\n\n')
        ctrl_img = [Image.open(p) for p in path]
    else:
        ctrl_img = Image.open(path)
    return ctrl_img

def save_control_imgs(control_img, output_path):
    if type(control_img) is list:
        for i in range(len(control_img)):
            control_img[i].save(os.path.join(output_path, f"ctrl_{i}.png"))
    else:
        control_img.save(os.path.join(output_path, f"ctrl_0.png"))

def prepare_target_prompts(target_prompts, num_controls):
    if len(target_prompts) == 1:
        target_prompts = target_prompts * num_controls
    else:
        assert len(target_prompts) == num_controls, f"Number of content prompts should equal to 1 or match the number of input conditional images. Num Prompts: {len(target_prompts)}, Num Control Conditions: {num_controls}"
    return target_prompts

def initialize_latents_controlnet(num_images_per_prompt, num_controls, dtype):
    latents = torch.randn(2, 4, 128, 128).to(dtype)
    latents[1] = torch.randn(1, 4, 128, 128).to(dtype)

    additional_latents = []
    if num_images_per_prompt > 1:
        for _ in range(num_images_per_prompt - 1):
            additional_latents.append(torch.randn(1, 4, 128, 128).to(dtype))
        latents = torch.cat([latents, torch.cat(additional_latents, dim=0)], dim=0)
    if num_controls > 1:
        target_latents = latents[1:]
        for _ in range(num_controls - 1):
            latents = torch.cat([latents, target_latents])
    
    return latents

def initialize_latents(num_prompts, num_images_per_prompt, dtype):
    latents = torch.randn(2, 4, 128, 128).to(dtype)
    latents[1:] = torch.randn(1, 4, 128, 128).to(dtype)

    additional_latents = []
    if num_images_per_prompt > 1:
        for _ in range(num_images_per_prompt - 1):
            additional_latents.append(torch.randn(1, 4, 128, 128).to(dtype))
        latents = torch.cat([latents, torch.cat(additional_latents, dim=0)], dim=0)
    for _ in range(2, num_prompts):
        latents = torch.cat([latents, latents[-num_images_per_prompt:]], dim=0)

    return latents