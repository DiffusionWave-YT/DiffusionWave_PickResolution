import nodes
import torch
import comfy.model_management

#Sub please "="
#https://www.youtube.com/@DiffusionWave

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

    RETURN_NAMES = ("INT Width", "INT Height", "FLOAT Width", "FLOAT Height", "INT Upscale Width", "INT Upscale Height", "FLOAT Upscale Width", "FLOAT Upscale Height", "Custom Upscaler")
    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT")
    FUNCTION = "generate_resolutions"
    CATEGORY = "Utilities"

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

NODE_CLASS_MAPPINGS = {
    "PickResolution_DiffusionWave 🌊": PickResolution_DiffusionWave,
}
