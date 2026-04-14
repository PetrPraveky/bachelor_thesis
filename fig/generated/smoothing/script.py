from PIL import Image

def upscale_image(input_path, output_path, size=(2000, 2000)):
    img = Image.open(input_path)
    img_upscaled = img.resize(size, resample=Image.Resampling.NEAREST)
    img_upscaled.save(output_path)

# 3 vstupní obrázky
upscale_image("binary_mask_eroded.png", "binary_mask_eroded_upscale.png")
upscale_image("binary_mask_grown.png", "binary_mask_grown_upscale.png")
upscale_image("binary_mask_smooth.png", "binary_mask_smooth_upscale.png")
upscale_image("binary_mask.png", "binary_mask_upscale.png")