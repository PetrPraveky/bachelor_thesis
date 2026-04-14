import numpy as np
from PIL import Image

def load_binary_mask(path):
    """
    Načte masku jako 8bit grayscale numpy pole.
    Očekává černobílý obrázek, ale funguje i pro grayscale.
    """
    img = Image.open(path).convert("L")
    return np.array(img)

def overlay_binary_on_grayed_base(
    base_mask_path,
    overlay_mask_path,
    output_path,
    gray_value=120,
    output_size=(2000, 2000)
):
    """
    Base mask:
        - černá zůstane černá
        - bílá se změní na šedou

    Overlay mask:
        - černá se ignoruje
        - bílá se vykreslí bíle přes base

    Výstup:
        - uloží se jako zvětšený obrázek nearest-neighbor interpolací
    """
    base = load_binary_mask(base_mask_path)
    overlay = load_binary_mask(overlay_mask_path)

    if base.shape != overlay.shape:
        raise ValueError(f"Mask sizes do not match: base={base.shape}, overlay={overlay.shape}")

    # binarizace pro jistotu
    base_fg = base > 127
    overlay_fg = overlay > 127

    # vytvoření výsledku
    out = np.zeros_like(base, dtype=np.uint8)

    # foreground base masky uděláme šedý
    out[base_fg] = gray_value

    # foreground overlaye dáme bíle
    out[overlay_fg] = 255

    # převod na PIL image
    out_img = Image.fromarray(out, mode="L")

    # upscale bez rozmazání
    out_img = out_img.resize(output_size, resample=Image.Resampling.NEAREST)

    out_img.save(output_path)

# --- použití ---
overlay_binary_on_grayed_base(
    "binary_mask_1.png",
    "skeleton_branches_1.png",
    "binary_mask_with_branches_1.png"
)

overlay_binary_on_grayed_base(
    "binary_mask_2.png",
    "skeleton_branches_2.png",
    "binary_mask_with_branches_2.png"
)