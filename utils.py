import numpy as np
import torch
import cv2
from PIL import Image

from transformers import DPTImageProcessor, DPTForDepthEstimation

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
feature_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")


def get_depth_ctrl(content_image_path, device):
    input_img = Image.open(content_image_path).resize((1024, 1024))
    depth_estimator = depth_estimator.to(device)
    feature_processor = feature_processor.to(device)
    control_img = calc_depth_map(input_img, feature_processor, depth_estimator)
    return control_img

def get_canny_ctrl(content_image_path, canny_params):
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
