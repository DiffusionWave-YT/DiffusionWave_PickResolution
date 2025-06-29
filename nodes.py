import nodes
import torch
import comfy.model_management
from PIL import Image
import numpy as np
import os
from datetime import datetime
from nodes import ImageBatch
from comfy.utils import ProgressBar


MAX_SEED_NUM = 999999999999

class PickResolution_DiffusionWave:
    @classmethod
    def INPUT_TYPES(s):
        resolutions = [
            "PICK RESOLUTION",
            "",
            "SQUARE",
            "512x512 (1:1)",
            "768x768 (1:1)",
            "896x896 (1:1)",            
            "1024x1024 (1:1)",
            "1080x1080 (1:1)",
            "1152x1152 (1:1)",
            "1280x1280 (1:1)",
            "",
            "VERTICAL",
            "512x768 (2:3)",
            "720x1080 (2:3)",
            "720x1280 (9:16)",
            "768x1024 (3:4)",
            "768x1152 (2:3)",
            "768x1280 (3:5)",
            "896x1152 (7:9)",
            "",
            "HORIZONTAL",
            "768x512 (3:2)",
            "1024x768 (4:3)",
            "1080x720 (3:2)",
            "1152x768 (3:2)",
            "1280x720 (16:9)",
            "1280x768 (5:3)",
            "1152x896 (9:7)",
        ]

        return {"required": {
            "BASE_RESOLUTION": (resolutions, ),
            "CUSTOM_UPSCALER": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.0000000001}),
            "SUM_EXTRA": ("INT", {"default": 0}),
        }}

    def generate_resolutions(self, BASE_RESOLUTION, CUSTOM_UPSCALER, SUM_EXTRA):
        dimensions = BASE_RESOLUTION.split(' ')[0]
        width, height = map(int, dimensions.split('x'))

        width_int = int((width // 8) * 8)
        height_int = int((height // 8) * 8)

        width_float = float(width_int)
        height_float = float(height_int)


        upscale_width_int = int(width * CUSTOM_UPSCALER) + SUM_EXTRA
        upscale_height_int = int(height * CUSTOM_UPSCALER) + SUM_EXTRA
        upscale_width_float = (width * CUSTOM_UPSCALER) + SUM_EXTRA
        upscale_height_float = (height * CUSTOM_UPSCALER) + SUM_EXTRA

        return (width_int, height_int, width_float, height_float, upscale_width_int, upscale_height_int, round(upscale_width_float, 10), round(upscale_height_float, 10), round(CUSTOM_UPSCALER, 10))

    RETURN_NAMES = ("INT Width", "INT Height", "FLOAT Width", "FLOAT Height", "INT Upscale Width", "INT Upscale Height", "FLOAT Upscale Width", "FLOAT Upscale Height", "Custom Upscaler")
    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT")
    FUNCTION = "generate_resolutions"
    CATEGORY = "Utilities"


class Int_PickResolution_DiffusionWave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "WIDTH": ("INT", {"default": 512, "min": 1}),
            "HEIGHT": ("INT", {"default": 512, "min": 1}),
            "CUSTOM_UPSCALER": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 1e-10}),
            "SUM_EXTRA": ("INT", {"default": 0}),
        }}

    RETURN_NAMES = (
        "INT Width", "INT Height", "FLOAT Width", "FLOAT Height",
        "INT Upscale Width", "INT Upscale Height",
        "FLOAT Upscale Width", "FLOAT Upscale Height",
        "Custom Upscaler"
    )
    RETURN_TYPES = (
        "INT", "INT", "FLOAT", "FLOAT",
        "INT", "INT", "FLOAT", "FLOAT",
        "FLOAT"
    )
    FUNCTION = "generate"
    CATEGORY = "Utilities"

    def generate(self, WIDTH, HEIGHT, CUSTOM_UPSCALER, SUM_EXTRA):
        width_int = int((WIDTH // 8) * 8)
        height_int = int((HEIGHT // 8) * 8)
        width_float = float(width_int)
        height_float = float(height_int)
        upscale_width_int = int(WIDTH * CUSTOM_UPSCALER) + SUM_EXTRA
        upscale_height_int = int(HEIGHT * CUSTOM_UPSCALER) + SUM_EXTRA
        upscale_width_float = (WIDTH * CUSTOM_UPSCALER) + SUM_EXTRA
        upscale_height_float = (HEIGHT * CUSTOM_UPSCALER) + SUM_EXTRA

        return (
            width_int, height_int, width_float, height_float,
            upscale_width_int, upscale_height_int,
            round(upscale_width_float, 10), round(upscale_height_float, 10),
            round(CUSTOM_UPSCALER, 10)
        )




class OverlayImages_DiffusionWave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("combined_image",)
    FUNCTION = "combine"
    CATEGORY = "Image/Composite"

    def combine(self, base_image, overlay_image):
        # Convert tensors to PIL
        base = Image.fromarray((base_image[0].cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        overlay = Image.fromarray((overlay_image[0].cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

        # Ensure same size
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)

        # Combine
        combined = Image.alpha_composite(base, overlay)

        # Back to tensor format
        combined_np = np.array(combined).astype(np.float32) / 255.0
        combined_tensor = torch.from_numpy(combined_np).unsqueeze(0)

        return (combined_tensor,)




class MergeImages_DiffusionWave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "overlay": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_image",)
    FUNCTION = "merge"
    CATEGORY = "Image/Composite"

    def merge(self, background, overlay):
        background = background.cpu().numpy()
        overlay = overlay.cpu().numpy()

        # Validar tamaños
        if len(background) != len(overlay):
            raise ValueError("Background and overlay must have the same batch size.")

        merged_images = []

        for i in range(len(background)):
            bg_np = (background[i] * 255).astype(np.uint8)
            ov_np = (overlay[i] * 255).astype(np.uint8)

            # Convertir a PIL
            bg_pil = Image.fromarray(bg_np).convert("RGBA")
            ov_pil = Image.fromarray(ov_np).convert("RGBA")

            # Clonar para evitar que el overlay se degrade
            bg_copy = bg_pil.copy()
            ov_copy = ov_pil.copy()
            mask = ov_copy.getchannel("A")

            # Redimensionar si es necesario
            if ov_copy.size != bg_copy.size:
                ov_copy = ov_copy.resize(bg_copy.size, Image.Resampling.LANCZOS)
                mask = mask.resize(bg_copy.size, Image.Resampling.LANCZOS)

            # Superponer respetando transparencia
            bg_copy.paste(ov_copy, (0, 0), mask)

            # Convertir de nuevo a tensor
            result_np = np.array(bg_copy).astype(np.float32) / 255.0
            merged_images.append(result_np)

        # Convertir a tensor final
        merged_batch = torch.tensor(np.stack(merged_images))
        return (merged_batch,)



class RemoveBackgroundByColor_DiffusionWave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "red": ("INT", {"default": 0, "min": 0, "max": 255}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255}),
                "threshold": ("INT", {"default": 30, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "remove_bg"
    CATEGORY = "Image/Alpha"

    def remove_bg(self, images, red, green, blue, threshold):
        output = []

        target_color = np.array([red, green, blue], dtype=np.uint8)

        for i in range(len(images)):
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            img_rgb = img_np[:, :, :3]
            alpha = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8) * 255

            # Crear máscara de similitud de color
            color_diff = np.linalg.norm(img_rgb - target_color, axis=2)
            alpha[color_diff < threshold] = 0  # Quitar fondo si es similar

            # Combinar canales RGBA
            rgba = np.dstack((img_rgb, alpha))
            rgba_norm = rgba.astype(np.float32) / 255.0
            output.append(rgba_norm)

        return (torch.tensor(np.stack(output)),)
    

class ResizeLongestSide_DiffusionWave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "resize_longest_to": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 512}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized",)
    FUNCTION = "resize"
    CATEGORY = "Image/Resize"

    def resize(self, images, resize_longest_to, divisible_by):
        resized_images = []

        for i in range(len(images)):
            img = images[i]
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            h, w = img_np.shape[:2]

            # Escala manteniendo proporción
            if w >= h:
                scale = resize_longest_to / w
            else:
                scale = resize_longest_to / h

            new_w = int(round(w * scale))
            new_h = int(round(h * scale))

            # Ajustar a múltiplos de divisible_by
            new_w = max(divisible_by, round(new_w / divisible_by) * divisible_by)
            new_h = max(divisible_by, round(new_h / divisible_by) * divisible_by)

            # Redimensionar con Lanczos
            img_pil = Image.fromarray(img_np).convert("RGBA")
            resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Normalizar a float
            resized_tensor = torch.from_numpy(np.array(resized).astype(np.float32) / 255.0)
            resized_images.append(resized_tensor)

        return (resized_images,)
    



class LoadImagesFromFolder_DiffusionWave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "load"
    CATEGORY = "Image/IO"

    def load(self, folder_path):
        if not os.path.exists(folder_path):
            raise ValueError(f"Path does not exist: {folder_path}")
        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")

        image_files = [
            f for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]

        loaded_images = []
        for filename in image_files:
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert("RGBA")
                img_np = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np)
                loaded_images.append(img_tensor)
            except Exception as e:
                print(f"Failed to load image {filename}: {e}")

        if not loaded_images:
            raise RuntimeError(f"No valid images loaded from folder: {folder_path}")

        return (loaded_images, len(loaded_images))





class ImageSimpleSaver_DiffusionWave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename": ("STRING", {"default": "image", "multiline": False}),
                "output_folder": ("STRING", {"default": "C:/Users/YourName/Desktop/output", "multiline": False}),
                "extension": (['png', 'jpeg', 'webp'],),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Image/IO"

    def save(self, images, filename, output_folder, extension):
        os.makedirs(output_folder, exist_ok=True)

        count = 0
        for idx, image in enumerate(images):
            np_img = (image.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(np.clip(np_img, 0, 255)).convert("RGBA")

            name = f"{filename}_{idx:03d}.{extension}"
            save_path = os.path.join(output_folder, name)

            if extension == 'png':
                img.save(save_path, optimize=True)
            else:
                img = img.convert("RGB")
                img.save(save_path, optimize=True, quality=95)

            count += 1

        print(f"✅ Saved {count} image(s) to: {output_folder}")
        return {}




class ImageBatchMulti_DiffusionWave:
    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {}
        for i in range(1, 21):
            optional_inputs[f"image_{i}"] = ("IMAGE",)
        return {
            "required": {},
            "optional": optional_inputs
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "Image/Batch"

    def combine(self, **kwargs):
        image_list = []

        for i in range(1, 21):
            key = f"image_{i}"
            img = kwargs.get(key)
            if img is None:
                continue

            # Si viene como batch
            if isinstance(img, torch.Tensor) and img.ndim == 4:
                image_list.extend(list(img))
            # Si viene como imagen única
            elif isinstance(img, torch.Tensor) and img.ndim == 3:
                image_list.append(img)
            # Si ya es lista
            elif isinstance(img, list):
                image_list.extend(img)

        if not image_list:
            raise ValueError("No valid image inputs received.")

        # Retornar lista de imágenes con resoluciones originales
        return (image_list,)


class PromptExpression_DiffusionWave:
    expressions = {
        1:  "(neutral face, blank expression, open mouth, soft gaze, emotionless look)",
        2:  "(neutral face, blank expression, closed mouth, straight lips, calm eyes)",
        3:  "(smile, open mouth, cheerful, bright eyes, joyful expression)",
        4:  "(smile, closed mouth, gentle expression, relaxed eyes, soft smile)",
        5:  "(sad, open mouth, trembling lips, downturned eyes, sorrowful expression)",
        6:  "(sad, closed mouth, downturned lips, melancholic gaze, subtle frown)",
        7:  "(crying, open mouth, sobbing face, distressed expression, watery eyes)",
        8:  "(crying, closed mouth, trembling lips, teary eyes, trying to hold back tears)",
        9:  "(scared, open mouth, wide eyes, anxious face, nervous sweat)",
        10: "(scared, closed mouth, widened eyes, tense lips, fearful gaze)",
        11: "(angry, open mouth, clenched teeth, furrowed brows, intense glare)",
        12: "(angry, closed mouth, narrowed eyes, tense expression, deep frown)",
        13: "(surprised, open mouth, wide eyes, raised eyebrows, stunned expression)",
        14: "(surprised, closed mouth, lifted eyebrows, wide eyes, silent shock)",
        15: "(thinking, open mouth, curious face, focused gaze, slightly tilted brows)",
        16: "(thinking, closed mouth, thoughtful expression, calm eyes, furrowed brow)",
        17: "(shy, open mouth, nervous eyes, looking down, timid face)",
        18: "(shy, closed mouth, downcast gaze, soft blush, bashful smile)",
        19: "(disgusted, open mouth, furrowed brows, displeased look)",
        20: "(disgusted, closed mouth, squinting eyes, unimpressed expression)",
        21: "(embarrassed, open mouth, heavy blush, awkward smile, flustered face)",
        22: "(embarrassed, closed mouth, tight lips, blushing face, nervous eyes)",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 22}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("expression_prompt",)
    FUNCTION = "generate"
    CATEGORY = "Prompt/Game"

    def generate(self, index):
        result = self.expressions.get(index, "")
        return (result,)
    


class Order_String_Tags_DiffusionWave:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"INPUT_STRING": ("STRING", {"multiline": True, "default": ""})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "order_tags"
    CATEGORY = "Text Processing"

    def order_tags(self, INPUT_STRING):
        tags = [tag.strip() for tag in INPUT_STRING.split(",") if tag.strip()]
        sorted_tags = ", ".join(sorted(tags))
        return (sorted_tags,)


class Blacklist_String_DiffusionWave:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "INPUT_STRING": ("STRING", {"multiline": True, "default": ""}),
                "BLACKLIST": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "filter_blacklist"
    CATEGORY = "Text Processing"

    def filter_blacklist(self, INPUT_STRING, BLACKLIST):
        input_tags = [tag.strip() for tag in INPUT_STRING.split(",") if tag.strip()]
        blacklist_tags = set(tag.strip() for tag in BLACKLIST.split(",") if tag.strip())
        
        filtered_tags = []
        for tag in input_tags:
            words = tag.split()
            words_filtered = [word for word in words if word not in blacklist_tags]
            if words_filtered:
                filtered_tags.append(" ".join(words_filtered))
        
        return (", ".join(sorted(filtered_tags)),)
    


MAX_SEED_NUM=999999999999

class Seed__DiffusionWave:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)    
    FUNCTION = "Seed__DiffusionWave_FUNC"
    CATEGORY = "Seed"

    def Seed__DiffusionWave_FUNC(self, seed=0, prompt=None, extra_pnginfo=None, my_unique_id=None):
        return seed,




NODE_CLASS_MAPPINGS = {
    "PickResolution_DiffusionWave 🌊": PickResolution_DiffusionWave,
    "Int_PickResolution_DiffusionWave 🌊": Int_PickResolution_DiffusionWave,
    "OverlayImages_DiffusionWave": OverlayImages_DiffusionWave,
    "MergeImages_DiffusionWave": MergeImages_DiffusionWave,
    "RemoveBackgroundByColor_DiffusionWave": RemoveBackgroundByColor_DiffusionWave,
    "ResizeLongestSide_DiffusionWave": ResizeLongestSide_DiffusionWave,
    "LoadImagesFromFolder_DiffusionWave": LoadImagesFromFolder_DiffusionWave,
    "ImageSimpleSaver_DiffusionWave": ImageSimpleSaver_DiffusionWave,
    "ImageBatchMulti_DiffusionWave": ImageBatchMulti_DiffusionWave,
    "PromptExpression_DiffusionWave 🌊": PromptExpression_DiffusionWave,
    "Order_String_Tags_DiffusionWave 🌊": Order_String_Tags_DiffusionWave,
    "Blacklist_String_DiffusionWave 🌊": Blacklist_String_DiffusionWave,
    "Seed__DiffusionWave 🌊": Seed__DiffusionWave,



}
