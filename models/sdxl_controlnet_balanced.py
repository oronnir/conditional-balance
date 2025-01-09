# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from diffusers import ControlNetModel
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version

from PIL import Image

T = torch.Tensor
TN = Union[T, None]

NUM_REF_IMAGES = 1

class StableDiffusionXLControlNetBalancedPipeline:
    """
    A class which applies adain handlers with balancing conditionals.
    """
    def __init__(self, pipeline, handler, controlnet_removal_amount) -> None:
        self.pipeline = pipeline
        self.handler = handler
        self.controlnet_removal_amount = controlnet_removal_amount


    def concat_zero_control(self, control_reisduel: T) -> T:
        b = control_reisduel.shape[0] // 2
        zerso_reisduel = torch.zeros_like(control_reisduel[0:1])
        return torch.cat([zerso_reisduel] + [control_reisduel[:b]] + [zerso_reisduel] + [control_reisduel[b:]])

    @torch.no_grad()
    def __call__(self,
        prompt: str | list[str] = None,
        prompt_2: str | list[str] | None = None,
        control_image: PipelineImageInput = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        latents: TN = None,
        prompt_embeds: TN = None,
        negative_prompt_embeds: TN = None,
        pooled_prompt_embeds: TN = None,
        negative_pooled_prompt_embeds: TN = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        controlnet_conditioning_scale: float | list[float] = 1.0,
        control_guidance_start: float | list[float] = 0.0,
        control_guidance_end: float | list[float] = 1.0,
        original_size: tuple[int, int] = None,
        crops_coords_top_left: tuple[int, int] = (0, 0),
        target_size: tuple[int, int] | None = None,
        negative_original_size: tuple[int, int] | None = None,
        negative_crops_coords_top_left: tuple[int, int] = (0, 0),
        negative_target_size:tuple[int, int] | None = None,
        clip_skip: int | None = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ) -> list[Image]:
        
        controlnet = self.pipeline.controlnet._orig_mod if is_compiled_module(self.pipeline.controlnet) else self.pipeline.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.pipeline.check_inputs(
            prompt=prompt,
            prompt_2=prompt_2,
            image=control_image,
            callback_steps=1,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self.pipeline._guidance_scale = guidance_scale

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipeline._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            control_image = self.pipeline.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=1,
                num_images_per_prompt=1,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=True,
                guess_mode=False,
            )
            height, width = control_image.shape[-2:]
            control_image = torch.stack([control_image[0]] * num_images_per_prompt + [control_image[1]] * num_images_per_prompt)  #TODO: NEW
        else:
            assert False

        # 5. Prepare timesteps
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipeline.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.pipeline.unet.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size= NUM_REF_IMAGES + num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
            
        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None

        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 7.2 Prepare added time ids & embeddings
        if isinstance(control_image, list):
            original_size = original_size or control_image[0].shape[-2:]
        else:
            original_size = original_size or control_image.shape[-2:]
        target_size = target_size or (height, width)

        add_text_embeds = pooled_prompt_embeds
        if self.pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.pipeline.text_encoder_2.config.projection_dim

        add_time_ids = self.pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.pipeline._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        prompt_embeds = torch.stack([prompt_embeds[0]] + [prompt_embeds[1]] * num_images_per_prompt)
        negative_prompt_embeds = torch.stack([negative_prompt_embeds[0]] + [negative_prompt_embeds[1]] * num_images_per_prompt)
        negative_pooled_prompt_embeds = torch.stack([negative_pooled_prompt_embeds[0]] + [negative_pooled_prompt_embeds[1]] * num_images_per_prompt)
        add_text_embeds = torch.stack([add_text_embeds[0]] + [add_text_embeds[1]] * num_images_per_prompt)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(NUM_REF_IMAGES + num_images_per_prompt, 1)
        batch_size = NUM_REF_IMAGES + num_images_per_prompt

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.pipeline.scheduler.order
        is_unet_compiled = is_compiled_module(self.pipeline.unet)
        is_controlnet_compiled = is_compiled_module(self.pipeline.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}  # TODO: NEW
        controlnet_prompt_embeds = torch.cat((prompt_embeds[NUM_REF_IMAGES:batch_size], prompt_embeds[NUM_REF_IMAGES:batch_size])) # TODO: NEW
        controlnet_added_cond_kwargs = {key: torch.cat((item[NUM_REF_IMAGES:batch_size,], item[NUM_REF_IMAGES:batch_size])) for key, item in added_cond_kwargs.items()} # TODO: NEW
        
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)           

                # controlnet(s) inference
                control_model_input = torch.cat((latent_model_input[NUM_REF_IMAGES:batch_size], latent_model_input[batch_size + NUM_REF_IMAGES:]))  # TODO: NEW

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    
                if cond_scale > 0: # TODO: NEW - if added and mid_block_res_sample is out of if and changed!
                    down_block_res_samples, mid_block_res_sample = self.pipeline.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=control_image,
                        conditioning_scale=cond_scale,
                        guess_mode=False,
                        added_cond_kwargs=controlnet_added_cond_kwargs,
                        return_dict=False,
                    )

                    if t < self.controlnet_removal_amount:  # This part controls geometric stylization
                        down_block_res_samples = [torch.zeros_like(down_block_res_sample) for down_block_res_sample in down_block_res_samples]
        
                    mid_block_res_sample = self.concat_zero_control(mid_block_res_sample)
                    down_block_res_samples =  [self.concat_zero_control(down_block_res_sample) for down_block_res_sample in down_block_res_samples]
                else:
                    mid_block_res_sample = down_block_res_samples = None

                # update layer stylization chice before unet call
                self.handler.update_adain_layers(t)

                # predict the noise residual
                noise_pred = self.pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0] 

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self.pipeline, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipeline.scheduler.order == 0): # TODO: NEW: added progress bar
                    progress_bar.update()
                
        # manually for max memory savings TODO: NEW: added part
        if self.pipeline.vae.dtype == torch.float16 and self.pipeline.vae.config.force_upcast:
            self.pipeline.upcast_vae()
            latents = latents.to(next(iter(self.pipeline.vae.post_quant_conv.parameters())).dtype)

        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.pipeline.vae.dtype == torch.float16 and self.pipeline.vae.config.force_upcast

        if needs_upcasting:
            self.pipeline.upcast_vae()
            latents = latents.to(next(iter(self.pipeline.vae.post_quant_conv.parameters())).dtype)

        image = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.pipeline.vae.to(dtype=torch.float16)
    
        if self.pipeline.watermark is not None:
            image = self.pipeline.watermark.apply_watermark(image)

        image = self.pipeline.image_processor.postprocess(image, output_type='pil')  # added outputs type

        # Offload all models
        self.pipeline.maybe_free_model_hooks()
        return image