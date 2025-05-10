import os, json
from shutil import copyfile
from style_gen_img_canny import generate_canny_control_image

focus_dataset_path = '/home/oron_nir/data/stylized_train_coco_aug_withneg_adjchange_merge.json'
with open(focus_dataset_path, 'r') as f:
    focus_data = json.load(f)
n_contents = 64
rep_content_prompts = [dict(image_path=os.path.basename(fd['image_path']), 
                        true_caption=fd['true_caption']) 
                        for fd in focus_data['style_labels'] if fd['split'] == 'Train']
added_images = set()
content_prompts = []
for fd in rep_content_prompts:
    if fd['image_path'] in added_images:
        continue
    added_images.add(fd['image_path'])
    content_prompts.append(fd)
content_prompts = content_prompts[:n_contents]
content_prompts = sorted(content_prompts, key=lambda x: x['image_path'])
# content_prompts = [
#     {"image_path": "COCO_train2014_000000205904.jpg", "true_caption": "Tourists dressed for cold weather talking to a policeman in a public area"},
#     {"image_path": "COCO_train2014_000000389704.jpg", "true_caption": "A man in blue shirt and orange vest next to a white boat"},
#     {"image_path": "COCO_train2014_000000572627.jpg", "true_caption": "A brown cat sitting on some steps looking down at a pair of brown shoes"},
#     {"image_path": "COCO_train2014_000000550670.jpg", "true_caption": "A green double decker bus is in an old city"},
#     ]

content_prompts_text = "\" \"".join([cp['true_caption'] for cp in content_prompts])
content_image_paths = [f"/home/oron_nir/data/Train/controls/{cp['image_path']}" for cp in content_prompts]

# create the controls directory if it does not exist and copy the images there
controls_dir = "/home/oron_nir/code/balance/conditional-balance/outputs/controls"
if not os.path.exists(controls_dir):
    os.makedirs(controls_dir)
for path in content_image_paths:
    if not os.path.exists(os.path.join(controls_dir, os.path.basename(path))):
        print(f"Copying {path} to {controls_dir}")
        copyfile(path, os.path.join(controls_dir, os.path.basename(path)))
        # os.system(f"cp {path} {controls_dir}")

art_painters = ["Henri Matisse", "Claude Monet", "Pablo Picasso", "Vincent van Gogh", "Andy Warhol", 
                "Paul Gauguin", "Diego Rivera", "Jean-Michel Basquiat", "Keith Haring", "Banksy"]
style_prompts = ["" if not p else f"in the style of {p}" for p in art_painters]

print(f'Generating images for {len(content_prompts)} content prompts and {len(style_prompts)} style prompts')

# validate that all paths exist
for path in content_image_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist")
batch_size = 16
for i, style_prompt in enumerate(style_prompts):
    for batch in range(0, len(content_prompts), batch_size):
        content_prompts_batch = content_prompts[batch:batch+batch_size]
        content_prompts_text = "\" \"".join([cp['true_caption'] for cp in content_prompts_batch])
        content_image_paths_batch = [f"/home/oron_nir/data/Train/controls/{cp['image_path']}" for cp in content_prompts_batch]
        controls_dir = "\" \"".join(content_image_paths_batch)
        ref_prompt = "A photograph" if style_prompt == "" else "A painting"

        # create the controls directory if it does not exist and copy the images there
        if os.path.exists(controls_dir):
            os.system(f"rm -rf {controls_dir}")
        os.makedirs(controls_dir, exist_ok=True)

        for path in content_image_paths_batch:
            if not os.path.exists(os.path.join(controls_dir, os.path.basename(path))):
                print(f"Copying {path} to {controls_dir}")
                copyfile(path, os.path.join(controls_dir, os.path.basename(path)))

        # generate the control image
        params = {
            'init_index': batch,
            'content_prompts': [cp['true_caption'] for cp in content_prompts_batch],
            'style_prompt': f"\"{style_prompt}\"",
            'reference_prompt': f"\"{ref_prompt}\"",
            'control_image_paths': content_image_paths_batch,
            'lambda_s': 0.57,
            'lambda_t': 0.8,
            'num_images_per_prompt': 1,
            'output_path': "outputs",
            'initialize_latents': True
        }
        os.chdir('/home/oron_nir/code/balance/conditional-balance/')
        generate_canny_control_image(**params)

        # run the script
#         script = f"python style_gen_img_canny.py --seed 42 \
# --content_prompts \"{content_prompts_text}\" \
# --style_prompt \"{style_prompt}\" \
# --reference_prompt \"{ref_prompt}\" \
# --control_image_path \"{controls_dir}\" \
# --lambda_s 0.57 \
# --lambda_t 0.8 \
# --num_images_per_prompt 1 \
# --output_path \"outputs\" \
# --initialize_latents"
    
    # --content_image_path \"assets/conditionals_dir_example\" \
    # print(f'Running script: {script}')
    # exceute the  --lambda_t 0.5
    # os.system(script)

# realistic script
# realcontent_prompts_text = "\" \"".join(["A photograph of " + cp['true_caption'] for cp in content_prompts])
# os.system(f'python style_gen_img_canny.py --seed 42 --content_prompts {realcontent_prompts_text} --style_prompt "" 
# --reference_prompt "An image" --control_image_path "/home/oron_nir/code/balance/conditional-balance/outputs/controls"
#  --lambda_s 0.0 --lambda_t 0.0 --num_images_per_prompt 1 --output_path "outputs" --initialize_latents')
for batch in range(0, len(content_prompts), batch_size):
    content_prompts_batch = content_prompts[batch:batch+batch_size]
    content_prompts_text = "\" \"".join([cp['true_caption'] for cp in content_prompts_batch])
    content_image_paths_batch = [f"/home/oron_nir/data/Train/controls/{cp['image_path']}" for cp in content_prompts_batch]
    controls_dir = "\" \"".join(content_image_paths_batch)
    ref_prompt = "A photograph"

    # generate the control image
    params = {
        'init_index': batch,
        'content_prompts': ["A photograph of " + cp['true_caption'] for cp in content_prompts_batch],
        'style_prompt': '""',
        'reference_prompt': "An image",
        'control_image_paths': content_image_paths_batch,
        'lambda_s': 0.0,
        'lambda_t': 0.0,
        'num_images_per_prompt': 1,
        'output_path': "outputs",
        'initialize_latents': True
    }
    os.chdir('/home/oron_nir/code/balance/conditional-balance/')
    generate_canny_control_image(**params)
